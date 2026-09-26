"""
MIN_PEAK_PROMINENCE_uV (=0.3) の妥当性を、実データで検証するスクリプト。

やること：
1. 現在の設定(0.3)で、実際に検出されたN35/P50/N95ピークのプロミネンス値の分布を確認する
   → 0.3という閾値が、実際の検出結果に対してどれくらい余裕があるか(境界ギリギリのケースが
      どれくらいあるか)を見る。
2. 閾値を 0.1〜0.6 の範囲で振り、検出結果(response_detected, low_confidence, 各振幅)が
   どれくらい変化するかを確認する(感度分析)。
"""
import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import detect_perg_waves as dpw


def load_all_traces(input_dir):
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    csv_files = [f for f in csv_files if os.path.basename(f) != "participants_info.csv"]
    all_traces = []
    for fp in csv_files:
        traces = dpw.load_perg_csv(fp)
        for ch, (times, volts) in traces.items():
            all_traces.append((os.path.basename(fp), ch, times, volts))
    return all_traces


def analyze_with_threshold(all_traces, threshold):
    """指定したプロミネンス閾値で、全チャンネルを検出し直す。"""
    dpw.MIN_PEAK_PROMINENCE_uV = threshold
    rows = []
    for fname, ch, times, volts in all_traces:
        result = dpw.detect_perg_waves(times, volts)
        row = {"file": fname, "channel": ch}
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


def prominence_distribution_at_baseline(all_traces, baseline=0.3):
    """baseline閾値で、実際に選ばれたP50/N35/N95ピークのプロミネンス値を集める。"""
    dpw.MIN_PEAK_PROMINENCE_uV = baseline
    p50_prominences = []
    n35_prominences = []
    n95_prominences = []

    for fname, ch, times, volts in all_traces:
        smoothed = dpw._moving_average(volts, dpw.SMOOTHING_WINDOW_SAMPLES)
        max_candidates = dpw._find_candidate_extrema(times, smoothed, mode="max")
        min_candidates = dpw._find_candidate_extrema(times, smoothed, mode="min")

        p50_candidate = dpw._pick_best_in_range(max_candidates, dpw.P50_PLAUSIBLE_RANGE_MSEC)
        if p50_candidate is None:
            continue
        p50_time, p50_index, p50_prom = p50_candidate
        p50_voltage = float(volts[p50_index])
        p50_prominences.append(p50_prom)

        n35_candidate = dpw._pick_best_in_range(min_candidates, dpw.N35_PLAUSIBLE_RANGE_MSEC, exclude_after=p50_time)
        if n35_candidate is not None:
            n35_prominences.append(n35_candidate[2])

        n95_valid = [c for c in min_candidates if float(volts[c[1]]) < p50_voltage]
        n95_candidate = dpw._pick_best_in_range(n95_valid, dpw.N95_PLAUSIBLE_RANGE_MSEC, exclude_before=p50_time)
        if n95_candidate is not None:
            n95_prominences.append(n95_candidate[2])

    return np.array(p50_prominences), np.array(n35_prominences), np.array(n95_prominences)


def summarize(name, arr):
    if len(arr) == 0:
        print(f"{name}: データなし")
        return
    print(f"{name}: n={len(arr)}, min={arr.min():.3f}, 5%tile={np.percentile(arr,5):.3f}, "
          f"中央値={np.median(arr):.3f}, 95%tile={np.percentile(arr,95):.3f}, max={arr.max():.3f}")
    # 閾値0.3にどれだけ近いケースがあるか
    near_threshold = np.sum((arr >= 0.3) & (arr < 0.4))
    print(f"  → プロミネンスが0.3〜0.4uV(閾値ギリギリ)のケース: {near_threshold}件 / {len(arr)}件")


def main():
    parser = argparse.ArgumentParser(description="MIN_PEAK_PROMINENCE_uVの妥当性を実データで検証する")
    parser.add_argument("--input", default="input", help="波形CSVを置いたフォルダ（デフォルト: input）")
    args = parser.parse_args()

    print(f"'{args.input}' フォルダから全チャンネルの波形を読み込み中...")
    all_traces = load_all_traces(args.input)
    print(f"読み込んだチャンネル数: {len(all_traces)}\n")

    print("=" * 70)
    print("【1】現在の閾値(0.3uV)で、実際に選ばれたピークのプロミネンス分布")
    print("=" * 70)
    p50_prom, n35_prom, n95_prom = prominence_distribution_at_baseline(all_traces, baseline=0.3)
    summarize("P50", p50_prom)
    summarize("N35", n35_prom)
    summarize("N95", n95_prom)

    print()
    print("=" * 70)
    print("【2】閾値を変化させた場合の、検出結果への影響(感度分析)")
    print("=" * 70)
    baseline_df = analyze_with_threshold(all_traces, 0.3)
    baseline_df = baseline_df.sort_values(["file", "channel"]).reset_index(drop=True)

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.5, 0.6]
    summary_rows = []
    for th in thresholds:
        df = analyze_with_threshold(all_traces, th)
        df = df.sort_values(["file", "channel"]).reset_index(drop=True)

        response_flips = (df["response_detected"] != baseline_df["response_detected"]).sum()
        low_conf_flips = (df["low_confidence"] != baseline_df["low_confidence"]).sum()

        p50_diff = (df["p50_amplitude_uV"] - baseline_df["p50_amplitude_uV"]).abs()
        n95_diff = (df["n95_amplitude_uV"] - baseline_df["n95_amplitude_uV"]).abs()
        p50_changed = (p50_diff > 0.05).sum()
        n95_changed = (n95_diff > 0.05).sum()

        summary_rows.append({
            "threshold": th,
            "response_detected_flips": response_flips,
            "low_confidence_flips": low_conf_flips,
            "p50_amplitude_changed(>0.05uV)": p50_changed,
            "n95_amplitude_changed(>0.05uV)": n95_changed,
        })
        print(f"閾値={th}: response_detected反転={response_flips}件, low_confidence反転={low_conf_flips}件, "
              f"P50振幅変化(>0.05uV)={p50_changed}件, N95振幅変化(>0.05uV)={n95_changed}件 "
              f"(全{len(df)}件中)")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(os.path.dirname(__file__), "prominence_sensitivity_result.csv"), index=False)
    print("\n結果を prominence_sensitivity_result.csv に保存しました。")

    # 手元に既存の検出結果CSV（perg_detection_result.csv）があれば、
    # このスクリプトのbaseline(0.3)出力と一致するかも確認する（無ければスキップ）
    existing_path = "perg_detection_result.csv"
    if os.path.exists(existing_path):
        print()
        print("=" * 70)
        print(f"【3】整合性確認：このスクリプトのbaseline(0.3)出力 vs 既存の{existing_path}")
        print("=" * 70)
        existing = pd.read_csv(existing_path)
        existing = existing.sort_values(["file", "channel"]).reset_index(drop=True)
        if len(existing) == len(baseline_df):
            p50_match = np.isclose(existing["p50_amplitude_uV"].fillna(-999), baseline_df["p50_amplitude_uV"].fillna(-999), atol=0.01).mean()
            print(f"P50振幅の一致率: {p50_match*100:.1f}%")
        else:
            print(f"件数が異なる（既存={len(existing)}件, 今回={len(baseline_df)}件）ため単純比較はスキップ")


if __name__ == "__main__":
    main()
