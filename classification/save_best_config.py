# save_best_configs_all.py
import os, json, glob, pandas as pd

BASE = "/home/kimgk3793/ray_results"
# 우선순위가 높은 메트릭부터 검사합니다.
METRIC_CANDIDATES = ["f1", "val_f1", "val_acc", "acc", "accuracy", "auc", "precision", "recall", "loss", "val_loss"]

def choose_metric(cols):
    cols = set(cols)
    for m in METRIC_CANDIDATES:
        if m in cols:
            return m
    return None

def is_min_metric(metric_name: str):
    # loss 류는 낮을수록 좋음
    return "loss" in metric_name.lower()

def load_params(trial_dir):
    # Ray가 trial dir에 저장하는 파라미터
    for name in ["params.json", "param.json", "config.json"]:
        p = os.path.join(trial_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                try:
                    return json.load(f)
                except:
                    pass
    # 마지막 수단: result.json 중 가장 마지막 라인
    rjson = os.path.join(trial_dir, "result.json")
    if os.path.exists(rjson):
        try:
            with open(rjson) as f:
                last = None
                for line in f:
                    if line.strip():
                        last = json.loads(line)
                if last and "config" in last:
                    return last["config"]
        except:
            pass
    return {}

def summarize_experiment(exp_dir):
    # 각 trial의 progress.csv 스캔
    trial_dirs = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
    best = None
    best_val = None
    best_metric = None

    # metric 후보를 찾기 위해 첫 progress.csv를 스캔
    metric_guess = None
    for t in trial_dirs:
        pcsv = os.path.join(t, "progress.csv")
        if os.path.exists(pcsv):
            try:
                df = pd.read_csv(pcsv)
                metric_guess = choose_metric(df.columns)
                if metric_guess:
                    break
            except Exception:
                continue

    if metric_guess is None:
        raise RuntimeError(f"[{os.path.basename(exp_dir)}] progress.csv에서 사용할 metric 컬럼을 찾지 못함.")

    minimize = is_min_metric(metric_guess)

    # 모든 트라이얼 중 베스트 찾기
    for t in trial_dirs:
        pcsv = os.path.join(t, "progress.csv")
        if not os.path.exists(pcsv):
            continue
        try:
            df = pd.read_csv(pcsv)
            if metric_guess not in df.columns:
                continue
            series = df[metric_guess].dropna()
            if len(series) == 0:
                continue
            val = series.min() if minimize else series.max()
            if (best_val is None) or (val < best_val if minimize else val > best_val):
                best_val = val
                best = t
        except Exception:
            continue

    if best is None:
        raise RuntimeError(f"[{os.path.basename(exp_dir)}] 베스트 트라이얼을 찾지 못함(메트릭: {metric_guess}).")

    # 해당 트라이얼의 파라미터 로드
    best_config = load_params(best)
    out_path = os.path.join(exp_dir, "best_config.json")
    with open(out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"✅ {os.path.basename(exp_dir)} | metric={metric_guess} | "
          f"{'min' if minimize else 'max'}={best_val:.6f} | saved -> {out_path}")
    return {
        "experiment": os.path.basename(exp_dir),
        "metric": metric_guess,
        "mode": "min" if minimize else "max",
        "best_value": best_val,
        "best_trial_dir": best,
        "best_config_path": out_path,
        "best_config": best_config,
    }

def main():
    exp_dirs = [d for d in glob.glob(os.path.join(BASE, "Tune_*_*")) if os.path.isdir(d)]
    if not exp_dirs:
        # 폴더 이름이 Tune_AP_0 형태라면 이 패턴으로도 잡힘
        exp_dirs = [d for d in glob.glob(os.path.join(BASE, "Tune_*")) if os.path.isdir(d)]

    results = []
    for exp in sorted(exp_dirs):
        try:
            results.append(summarize_experiment(exp))
        except Exception as e:
            print(f"⚠️ {os.path.basename(exp)}: {e}")

    # 전체 요약 저장
    summary_path = os.path.join(BASE, "best_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
