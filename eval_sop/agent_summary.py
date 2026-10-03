"""Aggregates agent_eval_*_{orig30,hard14}_t07_s{0,1,2}.json.

Per model x question set: mean +- std over the 3 seeds for
  tool  = all required tool calls made (right tool + right part_id),
  answer= pre-registered automatic grader passes,
  joint = tool AND answer (guards against answers that pass the lenient
          grader without the lookup, e.g. listing every supplier's rate),
and a 95% CI from a question-level bootstrap (resample questions, 2,000 reps,
seed 12345; each question's score averaged over seeds).

Usage: python -m eval_sop.agent_summary
"""
import glob
import json
import os
import re

import numpy as np

from eval_sop.common import RESULTS


def main():
    rows = []
    groups = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "agent_eval_*_t07_s*.json"))):
        m = re.search(r"agent_eval_(.+)_(orig30|hard14)_t07_s(\d)\.json", os.path.basename(f))
        d = json.load(open(f))
        groups.setdefault((d["summary"]["model"], m.group(2)), []).append(d)
    rng = np.random.default_rng(12345)
    out = []
    for (model, qset), runs in sorted(groups.items()):
        ids = [r["id"] for r in runs[0]["records"]]
        mat = {k: np.array([[float(bool(rec[k])) for rec in run["records"]] for run in runs])
               for k in ("tool_correct", "answer_correct")}
        mat["joint"] = mat["tool_correct"] * mat["answer_correct"]
        res = dict(model=model, question_set=qset, n_questions=len(ids), n_seeds=len(runs),
                   temperature=runs[0]["summary"]["temperature"], num_ctx=runs[0]["summary"]["num_ctx"],
                   forecasts=runs[0]["summary"]["forecasts"])
        idx = rng.integers(0, len(ids), size=(2000, len(ids)))
        for k, name in (("tool_correct", "tool"), ("answer_correct", "answer"), ("joint", "joint")):
            per_seed = mat[k].mean(axis=1)
            per_q = mat[k].mean(axis=0)
            bs = per_q[idx].mean(axis=1)
            res[name] = dict(mean=float(per_seed.mean()), seed_std=float(per_seed.std(ddof=1)),
                             per_seed=[float(x) for x in per_seed],
                             ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        res["per_question_joint_over_seeds"] = dict(zip(ids, [float(x) for x in mat["joint"].mean(axis=0)]))
        out.append(res)
        print(f"{model:12s} {qset:7s} n={len(ids)}x{len(runs)}  "
              + "  ".join(f"{n}={res[n]['mean']:.3f}±{res[n]['seed_std']:.3f} [{res[n]['ci95'][0]:.2f},{res[n]['ci95'][1]:.2f}]"
                          for n in ("tool", "answer", "joint")))
    with open(os.path.join(RESULTS, "agent_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
