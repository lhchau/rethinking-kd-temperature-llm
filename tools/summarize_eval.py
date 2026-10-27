"""Print ROUGE-L mean +- std over seeds for every benchmark under an eval directory.

    python tools/summarize_eval.py results/gpt2-base/fkl-tau3.0-seed10/best_rougeL/eval
"""
import glob
import json
import os
import statistics
import sys


def summarize(eval_dir):
    scores = {}
    for path in glob.glob(os.path.join(eval_dir, "**", "seed*", "metrics.json"), recursive=True):
        bench = os.path.relpath(os.path.dirname(os.path.dirname(path)), eval_dir)
        with open(path) as f:
            scores.setdefault(bench, []).append(json.load(f)["rougeL"])
    means = []
    for bench, values in sorted(scores.items()):
        means.append(statistics.mean(values))
        print(f"{bench:12s} {means[-1]:6.2f} +- {statistics.pstdev(values):.2f}  ({len(values)} seeds)")
    if means:
        print(f"{'average':12s} {statistics.mean(means):6.2f}")
    return scores


if __name__ == "__main__":
    summarize(sys.argv[1])
