# -*- coding: utf-8 -*-
"""
build_dataset.py
================
把 Camargo 数据集里所有被试的 ramp（斜坡）试次，整理成「每一步下坡一行」的表格，用来训练模型。

每一行 = 一条腿在稳定下坡时踩地一次（一个支撑期：脚跟着地 → 脚尖离地）
    输入 x：slope_deg（坡度）、speed_mps（速度）、step_length_m（步长）、stride_length_m（步幅）
    输出 y：peak_kfm（峰值 KFM）、kfm_impulse（KFM 冲量），以及除以体重后的版本

运行方法（PowerShell，在本文件所在的文件夹里）：
    python build_dataset.py

需要的库：pip install mat-io pandas numpy scipy matplotlib
"""

import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")                 # 只保存图片，不弹窗
import matplotlib.pyplot as plt

# =====================================================================
# 设置（一般只需要改这里）
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集文件夹：把下载的 AB06、AB07 …… 解压到这里。
# 程序会自动往下找所有 ramp 文件夹，多套几层文件夹也没关系。
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

# 结果输出到这里
OUT_DIR = os.path.join(SCRIPT_DIR, "dataset_out")

# 只跑某几个人（先测试时用），例如 ["AB06"]；None = 全部
SUBJECTS = None

ACTIVITY = "ramp"
TARGET_LABEL = "rampdescent"   # 只要稳定下坡段；过渡段 walk-rampdescent / rampdescent-walk 不要
MIN_LABEL_FRACTION = 0.9       # 一个支撑期里至少 90% 的时间点是 rampdescent，才算下坡步

# KFM = KFM_SIGN × knee_angle_moment
# 已用数据确认：这个数据集里膝关节角度负数 = 弯曲，所以力矩正数 = 伸膝（内力矩）= 外部屈膝力矩 KFM。
KFM_SIGN = +1

# 算峰值和冲量时，是否去掉支撑期最后一小段（另一只脚已经落地、这只脚正在离地）。
# 这段时间脚几乎不受力，力矩主要来自腿往前甩，不是体重压膝盖；坡度小时它会被误当成峰值。
# True：只算「脚跟着地 → 另一只脚着地」这一段；False：整个支撑期。
EXCLUDE_PRESWING = True

DT = 0.005                     # conditions / gc / id / markers 都是 200 Hz
MAX_STANCE_S = 1.5             # 支撑期超过 1.5 秒，多半是站着不动，不要
MAX_STRIDE_S = 2.5

# 体重（kg），按这个顺序找：
#   1. 这里手动填的，例如 {"AB06": 74.8, "AB07": 68.0}
#   2. 数据集自带的被试信息文件 SubjectInfo.mat（放在 dataset 文件夹里或本文件旁边都行）
#   3. 每个人 osimxml 文件夹里的 ABxx.osim 模型（把各身体部分的质量加起来）
BODY_MASS_KG = {}

MAKE_CHECK_PLOTS = True        # 每种坡度画一张检查图，用来人工确认支撑期和峰值找得对不对

NEEDED = ["conditions", "gcLeft", "gcRight", "id", "markers"]


# =====================================================================
# 读文件
# =====================================================================
def load_mat(path):
    """读一个 .mat 文件，返回字典（单独写成函数，方便测试时替换）"""
    from matio import load_from_mat
    return load_from_mat(path)


def get_table(mat):
    """从 load_from_mat 的结果里取出表格（DataFrame）"""
    for key in ("data", "labels"):
        if isinstance(mat.get(key), pd.DataFrame):
            return mat[key]
    for value in reversed(list(mat.values())):
        if isinstance(value, pd.DataFrame):
            return value
    raise ValueError("文件里没找到表格")


def clean_label(v):
    """把 ['rampdescent'] 这类写法变成干净的 'rampdescent'"""
    while isinstance(v, (list, tuple, np.ndarray)):
        flat = np.ravel(np.asarray(v, dtype=object))
        if flat.size == 0:
            return ""
        v = flat[0]
    return str(v).strip("[]'\" ")


def get_incline(cond_mat):
    """conditions 里的 rampIncline，单位：度"""
    value = cond_mat.get("rampIncline")
    if value is None:
        return np.nan
    return float(np.ravel(np.asarray(value, dtype=float))[0])


def align_to(base_time, df):
    """把表格对齐到同一条时间轴（这几个文件本来就对齐，这里只是保险）"""
    t = df["Header"].to_numpy(dtype=float)
    if len(t) == len(base_time) and np.allclose(t, base_time, atol=1e-4):
        return df.reset_index(drop=True)
    idx = np.clip(np.searchsorted(t, base_time), 1, len(t) - 1)
    idx = idx - ((base_time - t[idx - 1]) < (t[idx] - base_time))   # 取最近的时间点
    out = df.iloc[idx].reset_index(drop=True)
    out["Header"] = base_time
    return out


def load_trial(activity_dir, trial):
    """读一趟的 5 个文件，返回对齐后的表格和坡度"""
    mats = {name: load_mat(os.path.join(activity_dir, name, trial + ".mat")) for name in NEEDED}
    tables = {name: get_table(mat) for name, mat in mats.items()}
    base_time = tables["id"]["Header"].to_numpy(dtype=float)     # 以 id 的时间为准
    tables = {name: align_to(base_time, df) for name, df in tables.items()}
    labels = np.array([clean_label(v) for v in tables["conditions"]["Label"]])
    return tables, labels, get_incline(mats["conditions"])


# =====================================================================
# 找文件夹 / 体重
# =====================================================================
def guess_subject(dirpath, dataset_dir):
    """认出这个 ramp 文件夹是谁的，返回 (被试编号, 这个人的文件夹)"""
    parts = os.path.normpath(dirpath).split(os.sep)
    subj_pos = [i for i, p in enumerate(parts) if re.fullmatch(r"AB\d+", p, re.IGNORECASE)]
    if subj_pos:                                   # 路径里有 AB06 这样的文件夹
        i = subj_pos[-1]
        return parts[i].upper(), os.sep.join(parts[:i + 1])
    # 有的压缩包里没有 AB06 这一层：从旁边 osimxml 文件夹里的 AB06.osim 认出是谁
    d = os.path.normpath(dirpath)
    stop = os.path.normpath(dataset_dir)
    while d != stop and os.path.dirname(d) != d:
        parent = os.path.dirname(d)
        osim_dir = os.path.join(parent, "osimxml")
        if os.path.isdir(osim_dir):
            names = [m.group(1).upper() for f in os.listdir(osim_dir)
                     for m in [re.fullmatch(r"(AB\d+)\.osim", f, re.IGNORECASE)] if m]
            if len(names) == 1:
                return names[0], parent
        d = parent
    return None, None


def find_activity_dirs(dataset_dir):
    """找到所有 .../ABxx/日期/ramp 文件夹"""
    found, seen = [], set()
    for dirpath, dirnames, _ in os.walk(dataset_dir):
        if os.path.basename(dirpath) != ACTIVITY or "conditions" not in dirnames:
            continue
        subject, subject_dir = guess_subject(dirpath, dataset_dir)
        if subject is None:
            print(f"  [注意] 认不出这是哪个人的数据，跳过：{dirpath}\n"
                  f"         请把每个人的压缩包解压到单独的文件夹里，例如 dataset\\AB06")
            continue
        if SUBJECTS and subject not in SUBJECTS:
            continue
        key = (subject, os.path.relpath(dirpath, subject_dir))   # 同一个人同一天只算一次（防止解压了两份）
        if key in seen:
            print(f"  [注意] 重复的数据，跳过：{dirpath}")
            continue
        seen.add(key)
        found.append((subject, subject_dir, dirpath))
    return sorted(found)


def mass_from_osim(path):
    """把 .osim 模型里所有身体部分的质量加起来 = 体重（kg）"""
    total = 0.0
    for body in ET.parse(path).getroot().iter("Body"):
        m = body.find("mass")
        if m is not None and m.text:
            total += float(m.text)
    return total


def to_float(v):
    """把 74.84、[74.84]、[[74.84]] 这类写法都变成数字，变不了就返回 NaN"""
    try:
        return float(np.ravel(np.asarray(v, dtype=float))[0])
    except Exception:
        return np.nan


def load_subject_info():
    """读数据集自带的 SubjectInfo.mat，返回 ({被试: 体重kg}, 文件路径)。没找到返回 ({}, None)"""
    candidates = []
    for folder in (SCRIPT_DIR, DATASET_DIR):
        for dirpath, _, files in os.walk(folder):
            if os.path.normpath(dirpath).startswith(os.path.normpath(OUT_DIR)):
                continue
            candidates += [os.path.join(dirpath, f) for f in files
                           if re.fullmatch(r"subjectinfo\.(mat|csv)", f, re.IGNORECASE)]
            if folder == SCRIPT_DIR:
                break                                  # 本文件旁边只看一层
    for path in candidates:
        try:
            if path.lower().endswith(".csv"):
                tables = [pd.read_csv(path)]
            else:
                tables = [v for v in load_mat(path).values() if isinstance(v, pd.DataFrame)]
            for df in tables:
                df = df.reset_index()                  # MATLAB 表格的行名可能在 index 里
                subj_col = next((c for c in df.columns
                                 if df[c].map(clean_label).str.fullmatch(r"AB\d+", case=False).mean() > 0.5),
                                None)
                mass_col = next((c for c in df.columns
                                 if re.search(r"weight|mass", str(c), re.IGNORECASE)), None)
                if subj_col is None or mass_col is None:
                    print(f"  [注意] {path} 里没认出「被试」和「体重」两列，列名是：{list(df.columns)}")
                    continue
                info = {}
                for s, m in zip(df[subj_col].map(clean_label), df[mass_col].map(to_float)):
                    if 30 < m < 200:                       # 单位是 kg
                        info[s.upper()] = m
                if info:
                    return info, path
        except Exception as e:
            print(f"  [注意] 读取 {path} 出错：{e!r}")
    return {}, None


def get_body_mass(subject, subject_dir, subject_info, info_path):
    if subject in BODY_MASS_KG:
        return float(BODY_MASS_KG[subject]), "手动填写"
    if subject in subject_info:
        return subject_info[subject], os.path.basename(info_path)
    for dirpath, _, files in os.walk(subject_dir):
        for f in files:
            if f.lower() == f"{subject.lower()}.osim":     # 例如 osimxml/AB06.osim
                try:
                    m = mass_from_osim(os.path.join(dirpath, f))
                    if 30 < m < 200:
                        return m, f
                except Exception:
                    pass
    return np.nan, "没找到"


# =====================================================================
# 切步、算特征
# =====================================================================
def event_indices(pct):
    """gc 的百分比「跳回 0」的位置 = 事件发生的时刻（脚跟着地 / 脚尖离地）"""
    pct = np.asarray(pct, dtype=float)
    candidates = np.unique(np.concatenate([np.where(np.diff(pct) < -50)[0] + 1,
                                           np.where(pct == 0)[0]]))
    events = []
    for i in candidates:
        if not events or i - events[-1] > int(0.2 / DT):    # 0.2 秒内重复的只算一次
            events.append(int(i))
    return np.array(events, dtype=int)


def find_stances(gc):
    """返回每个支撑期：(脚跟着地行号, 脚尖离地行号, 下一次脚跟着地行号或 None)"""
    hs = event_indices(gc["HeelStrike"])
    to = event_indices(gc["ToeOff"])
    stances = []
    for k, h in enumerate(hs):
        next_hs = int(hs[k + 1]) if k + 1 < len(hs) else None
        later = to[to > h]
        if len(later) == 0:
            continue
        t_off = int(later[0])
        if next_hs is not None and t_off >= next_hs:     # 顺序不对，跳过
            continue
        if (t_off - h) * DT > MAX_STANCE_S:
            continue
        stances.append((int(h), t_off, next_hs))
    return stances


def horizontal(markers, name):
    """小球的水平坐标 (x, z)，单位 m。注意：这个数据集里 y 是竖直方向（高度）！"""
    xz = markers[[f"{name}_x", f"{name}_z"]].to_numpy(dtype=float) / 1000.0
    return pd.DataFrame(xz).interpolate(limit_area="inside").to_numpy()   # 补上短暂丢失的点


def step_features(h, t_off, next_hs, side, pelvis, heel):
    """一个支撑期的 x：速度、步长、步幅"""
    other = "r" if side == "l" else "l"
    out = {"stance_time_s": (t_off - h) * DT,
           "speed_mps": np.nan, "step_length_m": np.nan, "stride_length_m": np.nan}

    # 行走方向：支撑期内骨盆的水平位移方向
    d = pelvis[t_off] - pelvis[h]
    if np.all(np.isfinite(d)) and np.linalg.norm(d) > 1e-3:
        u = d / np.linalg.norm(d)
        # 步长 = 这只脚着地时，两只脚跟沿行走方向的距离
        out["step_length_m"] = abs(float(np.dot(heel[side][h] - heel[other][h], u)))

    # 速度、步幅：用从这次着地到同一只脚下一次着地的一个完整步态周期
    if next_hs is not None and (next_hs - h) * DT <= MAX_STRIDE_S:
        cycle_time = (next_hs - h) * DT
        out["stride_length_m"] = float(np.linalg.norm(heel[side][next_hs] - heel[side][h]))
        out["speed_mps"] = float(np.linalg.norm(pelvis[next_hs] - pelvis[h]) / cycle_time)
    return out


def kfm_outputs(stance_moment, n_load):
    """一个支撑期的 y：峰值 KFM 和 KFM 冲量（只积正的部分）
    stance_moment：整个支撑期的力矩；n_load：前多少个点参与计算（去掉最后的离地段）"""
    kfm = KFM_SIGN * stance_moment[:n_load]
    i_peak = int(np.argmax(kfm))
    return {"peak_kfm_Nm": float(kfm[i_peak]),
            "peak_time_pct": 100.0 * i_peak / (len(stance_moment) - 1),
            "load_end_pct": 100.0 * (n_load - 1) / (len(stance_moment) - 1),   # 计算范围在支撑期的哪里结束
            "peak_at_window_end": i_peak >= n_load - 2,     # True = 峰值正好卡在计算范围的末尾，需要检查
            "kfm_impulse_Nms": float(trapezoid(np.clip(kfm, 0, None), dx=DT))}


def load_window_end(h, t_off, other_hs):
    """计算峰值/冲量用的结束点：另一只脚着地的时刻（找不到就用脚尖离地）"""
    if EXCLUDE_PRESWING:
        c = other_hs[(other_hs > h) & (other_hs <= t_off)]
        if len(c):
            return int(c[0])
    return t_off


def process_trial(subject, trial, tables, labels, incline):
    """处理一趟，返回 (可用的步, 这趟里所有下坡支撑期的情况)"""
    markers = tables["markers"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pelvis = np.nanmean([horizontal(markers, "L_ASIS"), horizontal(markers, "R_ASIS")], axis=0)
    heel = {"l": horizontal(markers, "L_Heel"), "r": horizontal(markers, "R_Heel")}
    time = tables["id"]["Header"].to_numpy(dtype=float)

    heel_strikes = {"l": event_indices(tables["gcLeft"]["HeelStrike"]),
                    "r": event_indices(tables["gcRight"]["HeelStrike"])}

    rows, stance_log = [], []
    for side, gc_name in (("l", "gcLeft"), ("r", "gcRight")):
        other = "r" if side == "l" else "l"
        moment = tables["id"][f"knee_angle_{side}_moment"].to_numpy(dtype=float)
        for h, t_off, next_hs in find_stances(tables[gc_name]):
            if np.mean(labels[h:t_off + 1] == TARGET_LABEL) < MIN_LABEL_FRACTION:
                continue                                    # 不是稳定下坡
            segment = moment[h:t_off + 1]
            has_kfm = not np.isnan(segment).any()           # 有空值 = 没踩到测力板
            load_end = load_window_end(h, t_off, heel_strikes[other])
            stance_log.append((side, h, t_off, load_end, has_kfm))
            if not has_kfm:
                continue
            row = {"subject": subject, "trial": trial, "side": side, "slope_deg": incline}
            row.update(step_features(h, t_off, next_hs, side, pelvis, heel))
            row.update(kfm_outputs(segment, load_end - h + 1))
            row.update({"t_start": time[h], "t_end": time[t_off]})
            rows.append(row)
    return rows, stance_log


def plot_check(subject, trial, incline, tables, stance_log, path):
    """检查图：深绿 = 算峰值和冲量的那段，浅绿 = 去掉的离地段，灰色 = 下坡但没踩到测力板（丢掉）"""
    t = tables["id"]["Header"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, side, color in ((axes[0], "l", "tab:red"), (axes[1], "r", "tab:purple")):
        kfm = KFM_SIGN * tables["id"][f"knee_angle_{side}_moment"].to_numpy(dtype=float)
        ax.plot(t, kfm, color=color, lw=1.2, label=f"KFM {side.upper()} (Nm)")
        ax.axhline(0, color="k", lw=0.6)
        for s, h, t_off, load_end, ok in stance_log:
            if s != side:
                continue
            if not ok:
                ax.axvspan(t[h], t[t_off], color="0.75", alpha=0.4)
                continue
            ax.axvspan(t[h], t[load_end], color="tab:green", alpha=0.35)
            ax.axvspan(t[load_end], t[t_off], color="tab:green", alpha=0.12)
            i = h + int(np.argmax(kfm[h:load_end + 1]))
            ax.plot(t[i], kfm[i], "k^")
        ax.set_ylabel("KFM (Nm)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{subject} / {trial} / slope {incline}°   green = used (light = push-off, excluded), "
                 f"grey = no force plate, ▲ = peak KFM")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# 主程序
# =====================================================================
def main():
    # 防止在某些 Windows 电脑上因为输出编码问题报错（个别字符显示不了就用 ? 代替）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"找不到数据集文件夹：{DATASET_DIR}\n请把解压后的 AB06、AB07 …… 放进去，或修改 DATASET_DIR。")
    os.makedirs(os.path.join(OUT_DIR, "check_plots"), exist_ok=True)

    activity_dirs = find_activity_dirs(DATASET_DIR)
    if not activity_dirs:
        sys.exit(f"在 {DATASET_DIR} 里没找到任何 {ACTIVITY} 文件夹。")
    print(f"找到 {len(activity_dirs)} 个 {ACTIVITY} 文件夹")

    subject_info, info_path = load_subject_info()
    if info_path:
        print(f"体重来自：{info_path}（{len(subject_info)} 个人）\n")
    else:
        print("没找到 SubjectInfo.mat，体重改用每个人的 .osim 模型计算\n")

    all_rows, report, plotted_slopes = [], [], set()
    for subject, subject_dir, activity_dir in activity_dirs:
        mass, mass_source = get_body_mass(subject, subject_dir, subject_info, info_path)
        trials = sorted(os.path.splitext(f)[0]
                        for f in os.listdir(os.path.join(activity_dir, "conditions"))
                        if f.endswith(".mat"))
        n_desc, n_ok = 0, 0
        for k, trial in enumerate(trials, 1):
            print(f"\r{subject}: {k}/{len(trials)} {trial}          ", end="", flush=True)
            missing = [n for n in NEEDED
                       if not os.path.exists(os.path.join(activity_dir, n, trial + ".mat"))]
            if missing:
                report.append({"subject": subject, "trial": trial, "error": f"缺少 {missing}"})
                continue
            try:
                tables, labels, incline = load_trial(activity_dir, trial)
                rows, stance_log = process_trial(subject, trial, tables, labels, incline)
            except Exception as e:                              # 一趟出错不影响其他趟
                report.append({"subject": subject, "trial": trial, "error": repr(e)})
                continue
            for r in rows:
                r["body_mass_kg"] = mass
            all_rows += rows
            n_desc += len(stance_log)
            n_ok += len(rows)
            report.append({"subject": subject, "trial": trial, "slope_deg": incline,
                           "descent_stances": len(stance_log), "usable_steps": len(rows),
                           "error": ""})
            if rows and MAKE_CHECK_PLOTS and incline not in plotted_slopes:   # 每种坡度画一张
                plot_check(subject, trial, incline, tables, stance_log,
                           os.path.join(OUT_DIR, "check_plots", f"{subject}_{trial}.png"))
                plotted_slopes.add(incline)
        mass_text = f"{mass:.1f} kg（{mass_source}）" if np.isfinite(mass) else "体重：没找到！"
        print(f"\r{subject}: {len(trials)} 趟，下坡支撑期 {n_desc} 个，可用 {n_ok} 步，{mass_text}")

    report_df = pd.DataFrame(report)
    for c in ("descent_stances", "usable_steps"):
        if c in report_df:
            report_df[c] = report_df[c].astype("Int64")
    report_df.to_csv(os.path.join(OUT_DIR, "report.csv"), index=False, encoding="utf-8-sig")
    if not all_rows:
        sys.exit("\n没有得到任何可用的步，请看 dataset_out/report.csv 里的 error 列。")

    steps = pd.DataFrame(all_rows)
    steps["peak_kfm_Nm_per_kg"] = steps["peak_kfm_Nm"] / steps["body_mass_kg"]
    steps["kfm_impulse_Nms_per_kg"] = steps["kfm_impulse_Nms"] / steps["body_mass_kg"]
    cols = ["subject", "trial", "side", "slope_deg", "speed_mps", "step_length_m", "stride_length_m",
            "stance_time_s", "peak_kfm_Nm", "kfm_impulse_Nms", "body_mass_kg",
            "peak_kfm_Nm_per_kg", "kfm_impulse_Nms_per_kg", "peak_time_pct", "load_end_pct",
            "peak_at_window_end", "t_start", "t_end"]
    steps = steps[cols].round(4)
    steps.to_csv(os.path.join(OUT_DIR, "kfm_steps.csv"), index=False)

    # 另一种做法：每一趟的可用步取平均，一趟一行
    trial_means = (steps.groupby(["subject", "trial", "slope_deg"], as_index=False)
                   .agg(n_steps=("side", "size"),
                        **{c: (c, "mean") for c in cols[4:13]})).round(4)
    trial_means.to_csv(os.path.join(OUT_DIR, "kfm_trial_means.csv"), index=False)

    # ---------------- 总结和检查 ----------------
    print(f"\n[完成] 共 {len(steps)} 步，来自 {steps.subject.nunique()} 个人")
    print(f"   每步一行：{os.path.join(OUT_DIR, 'kfm_steps.csv')}")
    print(f"   每趟一行：{os.path.join(OUT_DIR, 'kfm_trial_means.csv')}")
    print(f"   检查图：  {os.path.join(OUT_DIR, 'check_plots')}")
    print("\n每个坡度有多少步：")
    print(steps.groupby("slope_deg").size().to_string())
    print("\n数值范围（中位数 / 最小 / 最大）：")
    show = ["speed_mps", "step_length_m", "stride_length_m", "stance_time_s",
            "peak_kfm_Nm", "kfm_impulse_Nms", "peak_kfm_Nm_per_kg"]
    print(steps[show].describe().loc[["50%", "min", "max"]].round(3).to_string())

    no_mass = sorted(steps.loc[steps.body_mass_kg.isna(), "subject"].unique())
    if no_mass:
        print(f"\n[注意] 这些人没找到体重，_per_kg 两列是空的：{no_mass}\n"
              f"  请在文件开头的 BODY_MASS_KG 里手动填写，例如 {{'{no_mass[0]}': 70.0}}")
    errors = [r for r in report if r.get("error")]
    if errors:
        print(f"\n[注意] 有 {len(errors)} 趟读取出错，详情看 report.csv 的 error 列")
    med = steps["speed_mps"].median()
    if not 0.4 < med < 1.8:
        print(f"\n[注意] 速度中位数 {med:.2f} m/s 不太正常（正常走路约 0.8–1.4），请检查坐标方向")


if __name__ == "__main__":
    main()
