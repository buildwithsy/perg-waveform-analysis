# ============================================================
# PERG波形自動解析：N35・P50・N95 検出
# ============================================================
#
# 対象データ：PERG-IOBA Dataset（PhysioNet, Open Data Commons Attribution License v1.0）
#   https://physionet.org/content/perg-ioba-dataset/1.0.0/
#
# 抽出項目はISCEV標準（PERGの国際臨床基準）に準拠する。
#   - ISCEV standard for clinical pattern electroretinography (PERG): 2012 update.
#     Doc Ophthalmol 126(1):1-7.
#
# 【設計方針】
# N35→P50→N95という3つの波形成分（陰性→陽性→陰性）を検出するにあたり、以下の方針で実装している：
#   - 時間窓を区切ってその中の極値（最小値/最大値）を探すアプローチ
#   - 検出結果を必ずQC画像（生波形＋検出位置のマーカー）で目視確認できるようにする設計
#   - load → detect → plot という関数分割
#
# 【抽出する項目とISCEV標準での定義】
#   - P50潜時：刺激反転開始(t=0)からP50頂点までの時間 [必須]
#   - P50振幅：N35の谷からP50の頂点までの高さ（peak-to-trough）[必須]
#       N35が不明瞭な場合は、ベースライン(t=0〜P50立ち上がり間の平均)からP50頂点までの高さで代用
#   - N95振幅：P50の頂点からN95の谷までの高さ（peak-to-trough）[必須]
#   - N95潜時：刺激反転開始からN95谷までの時間 [参考。N95頂点は幅広く測定精度が低いため必須ではない]
#   - N95:P50比：N95振幅 / P50振幅（内層網膜と外層網膜の機能バランスの指標）
#
# 【時間窓（探索窓）について】
# 固定時刻の値をそのまま読むのではなく、時間窓の中で実際の極値を探す。
# 以下の妥当範囲はISCEV標準の典型値（N35≈35ms, P50≈45-60ms, N95≈90-100ms）を参考に設定し、
# Normal群（正常）403件の実測分布（N35: 11.2-44.9ms, P50: 36.0-76.1ms, N95: 70.2-128.7ms）と
# 照らして妥当性を確認済み。境界付近で頭打ちになっている兆候は見られなかった。
#
# 【再現性について・2026-09-17追記】
# 同じコード・同じ入力データでも、numpy/pandas/scipyのバージョンが異なると、
# 浮動小数点計算のごく僅かな差により、プロミネンス閾値(MIN_PEAK_PROMINENCE_uV)や
# low_confidence閾値の境界ギリギりのケースで検出結果が変わることを実際に確認した
# （1354件中130件で数値のズレ、うち5件でlow_confidenceフラグの反転が発生）。
# これを受けて、以下を追加している：
#   1. 実行時に、動作中のライブラリバージョンが requirements.txt の想定と一致するか確認し、
#      一致しない場合は警告を表示する（check_environment_versions関数）。
#   2. 結果CSVと同時に、実際に使用したバージョン情報を記録したテキストファイルを出力する。
# 再現性を厳密に担保するには、requirements.txt に記載のバージョンで
# `pip install -r requirements.txt` を実行してから本スクリプトを動かすこと。

import matplotlib
matplotlib.use("Agg")  # ローカル実行時にディスプレイが無い環境でも画像保存できるようにする

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import sys
import glob
import platform
import argparse
import scipy
from datetime import datetime
from scipy.signal import find_peaks

# ------------------------------------------------------------
# 【再現性の確認】このコードを検証・確定した際に使用したライブラリバージョン。
# requirements.txt の内容と一致させること。
# 境界値付近の検出結果はバージョン差で変わりうることを実際に確認しているため、
# 単なる参考情報ではなく、結果の信頼性に関わる設定として扱う。
# ------------------------------------------------------------
EXPECTED_VERSIONS = {
    "numpy": "2.4.4",
    "pandas": "3.0.2",
    "scipy": "1.17.1",
    "matplotlib": "3.10.9",
}


def check_environment_versions():
    """
    実行中のライブラリバージョンを EXPECTED_VERSIONS と照合する。
    一致しない場合は、検出結果が想定と異なりうる旨を警告として表示する
    （実行は止めない。バージョン差で結果が変わることを実証済みのため、
    「知らずに違う結果を得てしまう」事態を防ぐことが目的）。
    戻り値：実際に使用されたバージョンの辞書。
    """
    actual_versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }

    mismatches = [
        f"  - {name}: 想定={expected} / 実際={actual_versions[name]}"
        for name, expected in EXPECTED_VERSIONS.items()
        if actual_versions[name] != expected
    ]

    if mismatches:
        print("=" * 60)
        print("[警告] ライブラリのバージョンが想定と異なります。")
        print("境界値付近のケースでは、検出結果が過去の実行と一致しない可能性があります。")
        print("\n".join(mismatches))
        print("再現性を確保するには: pip install -r requirements.txt")
        print("=" * 60)
    else:
        print("[OK] ライブラリバージョンは想定通りです。再現性のある結果が期待できます。")

    return actual_versions


def write_environment_log(actual_versions, result_csv_path):
    """
    実行に使用した環境情報（ライブラリバージョン、Pythonバージョン、実行日時）を、
    結果CSVと同じ場所にテキストファイルとして記録する。
    「このCSVはどの環境で生成されたか」を後から追跡できるようにするため。
    """
    log_path = os.path.splitext(result_csv_path)[0] + "_environment.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"実行日時: {datetime.now().isoformat()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"OS: {platform.platform()}\n")
        for name, version in actual_versions.items():
            expected = EXPECTED_VERSIONS.get(name, "(想定なし)")
            match = "一致" if version == expected else "★不一致"
            f.write(f"{name}: {version} (想定: {expected}, {match})\n")
    return log_path

# ------------------------------------------------------------
# 【設定】各成分が現れる「妥当な時間範囲」（ms）。
# 成分ごとに固定の探索窓を設け、その窓内の最大値/最小値を機械的に返す方式だと、
# 波形が平坦・ノイズのみの場合でも意味のない値を返してしまう問題があった。
# そのため、まず波形全体から本物らしき極値（周囲より十分高い/低い点）を検出し、
# その中からN35(陰性)→P50(陽性)→N95(陰性)の順序制約で絞り込む方式にしている。
# 以下の妥当範囲は、「この範囲の外にある極値は候補から除外する」ための、
# 厳密な探索窓ではなくゆるめの目安として使う。
# ------------------------------------------------------------
N35_PLAUSIBLE_RANGE_MSEC = (10.0, 45.0)
P50_PLAUSIBLE_RANGE_MSEC = (35.0, 80.0)
N95_PLAUSIBLE_RANGE_MSEC = (70.0, 130.0)

# 極値とみなすための最小プロミネンス（周囲との高さの差、uV単位）。
# ノイズによる小さなギザギザを「本物の山・谷」として拾わないためのフィルタ。
# 2026-09-26検証済み：全1354チャンネルで、実際に検出されたP50/N35/N95ピークの
# プロミネンス分布は中央値1.4〜2.7uVで、閾値0.3ギリギリ(0.3-0.4uV)のケースは
# 1.4〜2.5%程度と少数。閾値を0.1〜0.6で振る感度分析でも、0.3から離れるほど
# 検出結果の変化件数が増える(0.3付近が局所的に安定)ことを確認済み。
# 詳細：意思決定ログ2026-09-26のエントリ、prominence_sensitivity_result.csv参照。
MIN_PEAK_PROMINENCE_uV = 0.3

# P50振幅がこの値を下回ったら low_confidence フラグを立てる。
# 根拠：ISCEV標準のP50正常範囲下限(2.0uV)の半分を暫定的な境界とした。
# 波形がほぼ平坦・無反応に近い症例では、CSVの数値が0.1uV刻みに丸められていることに由来する
# 量子化ノイズが、本物の小さなピークのように誤検出されることがあったため導入した。
# 暫定値（未検証）。
LOW_CONFIDENCE_P50_AMPLITUDE_THRESHOLD_uV = 1.0

# 波形のノイズ除去用の移動平均の窓幅（サンプル数）。極値検出の前処理としてのみ使い、
# 振幅の読み取りには生データを使う（位置検出と振幅読み取りを分ける設計）。
SMOOTHING_WINDOW_SAMPLES = 5


def load_participants_info(participants_info_path):
    """
    participants_info.csv を読み込み、id_record(例:"0001") -> diagnosis1 の辞書を返す。
    ファイルが存在しない場合は空の辞書を返す（診断名なしで処理を続行できるようにする）。
    """
    if not participants_info_path or not os.path.exists(participants_info_path):
        return {}

    info_df = pd.read_csv(participants_info_path, dtype={"id_record": str})
    info_df["id_record"] = info_df["id_record"].str.zfill(4)
    return dict(zip(info_df["id_record"], info_df["diagnosis1"]))


def guess_id_record_from_filename(file_path):
    """ファイル名の先頭にある4桁の数字を id_record として抜き出す。"""
    basename = os.path.basename(file_path)
    match = re.match(r"(\d{4})", basename)
    return match.group(1) if match else None


def load_perg_csv(file_path):
    """
    PERG-IOBA形式のCSV（TIME_1, RE_1, LE_1, ...）を読み込み、
    チャンネルごとの時刻(ms)・電圧(uV)の波形を辞書で返す。
    """
    df = pd.read_csv(file_path)
    time_cols = [c for c in df.columns if c.startswith("TIME")]

    traces = {}
    for time_col in time_cols:
        suffix = time_col.replace("TIME", "")
        for eye_prefix in ["RE", "LE"]:
            signal_col = f"{eye_prefix}{suffix}"
            if signal_col not in df.columns:
                continue

            valid = df[[time_col, signal_col]].dropna()
            if valid.empty:
                continue

            times_dt = pd.to_datetime(valid[time_col])
            t0 = times_dt.iloc[0]
            times_ms = (times_dt - t0).dt.total_seconds().to_numpy() * 1000.0
            volts_uV = valid[signal_col].to_numpy(dtype=float)

            traces[signal_col] = (times_ms, volts_uV)

    return traces


def _moving_average(volts, window_samples):
    """極値検出用の軽い平滑化（移動平均）。振幅の読み取りには使わない。"""
    if window_samples <= 1:
        return volts.copy()
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(volts, kernel, mode="same")


def _find_candidate_extrema(times, smoothed_volts, mode):
    """
    平滑化した波形から、本物らしき極値（周囲より十分高い/低い点）を候補として抽出する。
    mode="max" なら山（正のピーク）、mode="min" なら谷（負のピーク）の候補を返す。
    戻り値：[(time_ms, index), ...] のリスト（プロミネンス順ではなく時間順）。
    """
    signal_for_peaks = smoothed_volts if mode == "max" else -smoothed_volts
    peak_indices, properties = find_peaks(signal_for_peaks, prominence=MIN_PEAK_PROMINENCE_uV)
    return [(float(times[i]), i, float(properties["prominences"][j]))
            for j, i in enumerate(peak_indices)]


def _pick_best_in_range(candidates, time_range, exclude_before=None, exclude_after=None):
    """
    候補極値の中から、指定した時間範囲・条件を満たすものの中で
    最もプロミネンス（周囲との高さの差）が大きいものを選ぶ。
    条件を満たす候補が無ければ None を返す。
    """
    start_ms, end_ms = time_range
    filtered = [c for c in candidates if start_ms <= c[0] <= end_ms]
    if exclude_before is not None:
        filtered = [c for c in filtered if c[0] > exclude_before]
    if exclude_after is not None:
        filtered = [c for c in filtered if c[0] < exclude_after]

    if not filtered:
        return None
    return max(filtered, key=lambda c: c[2])  # プロミネンス最大のものを選ぶ


def detect_perg_waves(times, volts):
    """
    1本のPERG波形（時刻ms・電圧uV）から、N35・P50・N95を検出する。

    波形全体から本物らしき極値を検出し、N35(陰性)→P50(陽性)→N95(陰性)という
    順序制約で絞り込む方式を採用している。固定の探索窓の中の最大値/最小値を
    機械的に返す方式だと、波形が平坦・ノイズのみの場合でも意味のない値を
    返してしまうため、この方式にしている。
    """
    smoothed = _moving_average(volts, SMOOTHING_WINDOW_SAMPLES)

    max_candidates = _find_candidate_extrema(times, smoothed, mode="max")
    min_candidates = _find_candidate_extrema(times, smoothed, mode="min")

    # --- P50：妥当な時間範囲内で最もプロミネンスが大きい正のピーク ---
    p50_candidate = _pick_best_in_range(max_candidates, P50_PLAUSIBLE_RANGE_MSEC)

    if p50_candidate is None:
        # P50すら見つからない = 明確な反応が無い（無反応に近い）可能性が高い
        return {
            "n35_time_msec": None, "n35_voltage_uV": None,
            "p50_time_msec": None, "p50_voltage_uV": None,
            "p50_amplitude_uV": np.nan, "p50_amplitude_method": "P50未検出のため計算不可",
            "n95_time_msec": None, "n95_voltage_uV": None,
            "n95_amplitude_uV": np.nan, "n95_p50_ratio": np.nan,
            "response_detected": False,
            "low_confidence": True,
            "detection_note": f"波形全体を探しても、妥当な時間範囲({P50_PLAUSIBLE_RANGE_MSEC[0]:.0f}-{P50_PLAUSIBLE_RANGE_MSEC[1]:.0f}ms)に本物らしき正のピークが見つからなかった。無反応、または信号がノイズに埋もれている可能性がある。",
        }

    p50_time, p50_index, _ = p50_candidate
    p50_voltage = float(volts[p50_index])  # 振幅は生データから読む

    # --- N35：P50より前で、妥当な時間範囲内の負のピーク ---
    n35_candidate = _pick_best_in_range(min_candidates, N35_PLAUSIBLE_RANGE_MSEC, exclude_after=p50_time)
    if n35_candidate is not None:
        n35_time, n35_index, _ = n35_candidate
        n35_voltage = float(volts[n35_index])
    else:
        n35_time, n35_voltage = None, None

    # --- N95：P50より後で、妥当な時間範囲内の負のピーク ---
    # N95は単に「局所的な谷」であるだけでなく、「P50の電圧より低いこと」も必須条件にする。
    # 全体的に右肩上がりにドリフトする平坦な波形（明確なPERG反応の無い波形）では、
    # 局所的には谷でも絶対値としてはP50より高い点を拾ってしまい、N95振幅が定義上
    # ありえない負の値になる矛盾が起きることがあるため、この制約を追加している。
    n95_valid_candidates = [c for c in min_candidates if float(volts[c[1]]) < p50_voltage]
    n95_candidate = _pick_best_in_range(n95_valid_candidates, N95_PLAUSIBLE_RANGE_MSEC, exclude_before=p50_time)
    if n95_candidate is not None:
        n95_time, n95_index, _ = n95_candidate
        n95_voltage = float(volts[n95_index])
    else:
        n95_time, n95_voltage = None, None

    # --- P50振幅：N35の谷からP50の頂点までの高さ（ISCEV標準）。N35不明瞭ならベースライン代用 ---
    if n35_voltage is not None:
        p50_amplitude_uV = p50_voltage - n35_voltage
        p50_amplitude_method = "N35谷からの高さ"
    else:
        baseline_mask = (times >= 0.0) & (times <= p50_time)
        baseline_uV = float(np.mean(volts[baseline_mask])) if np.any(baseline_mask) else np.nan
        p50_amplitude_uV = p50_voltage - baseline_uV
        p50_amplitude_method = "ベースライン(N35不明瞭のため代用)からの高さ"

    # --- N95振幅：P50の頂点からN95の谷までの高さ（ISCEV標準） ---
    if n95_voltage is not None:
        n95_amplitude_uV = p50_voltage - n95_voltage
    else:
        n95_amplitude_uV = np.nan

    if not np.isnan(n95_amplitude_uV) and p50_amplitude_uV not in (0, None) and not np.isnan(p50_amplitude_uV):
        n95_p50_ratio = n95_amplitude_uV / p50_amplitude_uV
    else:
        n95_p50_ratio = np.nan

    detection_note = ""
    if n35_voltage is None:
        detection_note += f"N35が妥当な範囲({N35_PLAUSIBLE_RANGE_MSEC[0]:.0f}-{N35_PLAUSIBLE_RANGE_MSEC[1]:.0f}ms)に見つからなかった。"
    if n95_voltage is None:
        detection_note += f"N95が妥当な範囲({N95_PLAUSIBLE_RANGE_MSEC[0]:.0f}-{N95_PLAUSIBLE_RANGE_MSEC[1]:.0f}ms)に見つからなかった。"

    # P50振幅が閾値未満なら信頼度低フラグを立てる。
    # 量子化ノイズ由来と思われる小さなプラトーを、本物のP50として扱ってしまうのを防ぐため。
    low_confidence = p50_amplitude_uV < LOW_CONFIDENCE_P50_AMPLITUDE_THRESHOLD_uV
    if low_confidence:
        detection_note += f"P50振幅が{LOW_CONFIDENCE_P50_AMPLITUDE_THRESHOLD_uV}uV未満(信頼度低、量子化ノイズの可能性)。"

    return {
        "n35_time_msec": n35_time,
        "n35_voltage_uV": n35_voltage,
        "p50_time_msec": p50_time,
        "p50_voltage_uV": p50_voltage,
        "p50_amplitude_uV": p50_amplitude_uV,
        "p50_amplitude_method": p50_amplitude_method,
        "n95_time_msec": n95_time,
        "n95_voltage_uV": n95_voltage,
        "n95_amplitude_uV": n95_amplitude_uV,
        "n95_p50_ratio": n95_p50_ratio,
        "response_detected": True,
        "low_confidence": low_confidence,
        "detection_note": detection_note,
    }


def save_qc_plot(channel_label, times, volts, result, save_dir):
    """
    1チャンネル分のQC(目視確認用)グラフを画像として保存する。
    生波形の上に、検出したN35・P50・N95の位置を印として重ねて描く。
    （検出結果は必ず目視確認できるようにする設計）
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 4.5))
    plt.plot(times, volts, color="tab:blue", linewidth=1.5, label="PERG signal")
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # 「妥当な時間範囲」を薄い背景色で示す（以前のような厳密な探索窓ではなく、目安として）
    plt.axvspan(*N35_PLAUSIBLE_RANGE_MSEC, color="purple", alpha=0.07)
    plt.axvspan(*P50_PLAUSIBLE_RANGE_MSEC, color="orange", alpha=0.07)
    plt.axvspan(*N95_PLAUSIBLE_RANGE_MSEC, color="green", alpha=0.07)

    if result["n35_time_msec"] is not None:
        plt.scatter([result["n35_time_msec"]], [result["n35_voltage_uV"]],
                    color="purple", zorder=5, label=f"N35 ({result['n35_time_msec']:.1f}ms)")
    if result["p50_time_msec"] is not None:
        plt.scatter([result["p50_time_msec"]], [result["p50_voltage_uV"]],
                    color="orange", marker="^", zorder=5, label=f"P50 ({result['p50_time_msec']:.1f}ms)")
    if result["n95_time_msec"] is not None:
        plt.scatter([result["n95_time_msec"]], [result["n95_voltage_uV"]],
                    color="green", marker="v", zorder=5, label=f"N95 ({result['n95_time_msec']:.1f}ms)")

    # 検出不能・注記がある場合は、QC画像上にも赤字で明示する（見た目だけで異常に気づけるように）
    # ※matplotlibの標準フォントは日本語グリフを含まないため、画像内の注記は英語表記にする
    if not result.get("response_detected", True):
        plt.text(0.5, 0.95, "[NOT DETECTED] No clear P50 peak found", color="red", fontsize=10,
                  ha="center", va="top", transform=plt.gca().transAxes,
                  bbox=dict(facecolor="white", edgecolor="red", alpha=0.8))
    elif result.get("low_confidence"):
        plt.text(0.5, 0.95, f"[LOW CONFIDENCE] P50 amplitude < {LOW_CONFIDENCE_P50_AMPLITUDE_THRESHOLD_uV}uV (possible quantization noise)",
                  color="red", fontsize=8, ha="center", va="top", transform=plt.gca().transAxes,
                  bbox=dict(facecolor="white", edgecolor="red", alpha=0.8))
    elif result.get("detection_note"):
        note_en = []
        if result["n35_time_msec"] is None:
            note_en.append("N35 not found in plausible range")
        if result["n95_time_msec"] is None:
            note_en.append("N95 not found in plausible range")
        plt.text(0.5, 0.95, "[NOTE] " + " / ".join(note_en), color="darkorange", fontsize=8,
                  ha="center", va="top", transform=plt.gca().transAxes,
                  bbox=dict(facecolor="white", edgecolor="darkorange", alpha=0.8))

    # 症例間で見比べられるよう、軸は全症例で固定する。
    # 暫定的な設定であり、極端な振幅の症例が出てきたら見直す。
    plt.xlim(0, 150)
    plt.ylim(-10, 10)
    plt.xlabel("Time (msec)")
    plt.ylabel("Voltage (uV)")
    plt.title(f"PERG waveform QC : {channel_label}")
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"QC_{channel_label}.png")
    plt.savefig(save_path, dpi=130)
    plt.close()

    return save_path


def analyze_perg_file(file_path, output_dir, label=""):
    """
    1つのPERG CSVファイルを読み込み、全チャンネル（右眼・左眼）を解析して
    結果の一覧（リスト）とQC画像を出力する。
    """
    traces = load_perg_csv(file_path)
    results = []

    for channel_name, (times, volts) in traces.items():
        result = detect_perg_waves(times, volts)
        channel_label = f"{os.path.splitext(os.path.basename(file_path))[0]}_{channel_name}"
        if label:
            channel_label = f"{channel_label}_{label}"

        qc_path = save_qc_plot(channel_label, times, volts, result, output_dir)
        print(f"[完了] {channel_label} のQC画像を保存: {qc_path}")

        row = {"file": os.path.basename(file_path), "channel": channel_name}
        row.update(result)
        results.append(row)

    return pd.DataFrame(results)


def analyze_all_csv_in_dir(input_dir, output_dir, participants_info_path=None):
    """
    input_dir 直下にある波形CSV（participants_info.csv自身は除く）をすべて解析する。
    participants_info_path を指定すれば、id_recordから診断名(diagnosis1)を自動取得し、
    QC画像のファイル名やラベルに反映する。

    ファイルが増えるたびにコードを書き換える必要はなく、input_dirにCSVを置くだけでよい。
    """
    diagnosis_lookup = load_participants_info(participants_info_path)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    csv_files = [f for f in csv_files if os.path.basename(f) != "participants_info.csv"]

    if not csv_files:
        print(f"[警告] {input_dir} に波形CSVが見つかりませんでした。")
        return pd.DataFrame()

    all_results = []
    for file_path in csv_files:
        id_record = guess_id_record_from_filename(file_path)
        label = diagnosis_lookup.get(id_record, "") if id_record else ""
        # スペースなど、ファイル名に使えない文字をアンダースコアに置換
        safe_label = re.sub(r"[^\w\-]", "_", label) if label else ""

        df = analyze_perg_file(file_path, output_dir, label=safe_label)
        df["id_record"] = id_record
        df["diagnosis1"] = label
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PERG波形からN35/P50/N95を自動検出する")
    parser.add_argument("--input", default="input", help="波形CSVを置いたフォルダ（デフォルト: input）")
    parser.add_argument("--output", default="qc_images", help="QC画像の出力先フォルダ（デフォルト: qc_images）")
    parser.add_argument("--participants-info", default=None,
                         help="participants_info.csvのパス（省略時は--input直下を自動で探す）")
    parser.add_argument("--result-csv", default="perg_detection_result.csv",
                         help="検出結果を書き出すCSVファイル名")
    args = parser.parse_args()

    # 実行前に、ライブラリバージョンが想定通りか確認する（再現性の担保）
    actual_versions = check_environment_versions()

    os.makedirs(args.input, exist_ok=True)

    participants_info_path = args.participants_info or os.path.join(args.input, "participants_info.csv")

    result_df = analyze_all_csv_in_dir(
        input_dir=args.input,
        output_dir=args.output,
        participants_info_path=participants_info_path,
    )

    if not result_df.empty:
        print(f"\n=== 検出結果（{len(result_df)}件） ===")
        result_df.to_csv(args.result_csv, index=False)
        print(f"結果を {args.result_csv} に保存しました。")
        print(f"QC画像を {args.output}/ に保存しました。")

        env_log_path = write_environment_log(actual_versions, args.result_csv)
        print(f"実行環境の記録を {env_log_path} に保存しました。")
