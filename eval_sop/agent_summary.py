"""Aggregates agent_eval_*_{orig30,hard14}_t07_s{0,1,2}.json.

Per model x question set: mean +- std over the 3 seeds for
  tool  = all required tool calls made (right tool + right part_id),
  answer= pre-registered automatic grader passes,
  joint = tool AND answer (guards against answers that pass the lenient
          grader without the lookup, e.g. listing every supplier's rate),
and a 95% CI from a TEMPLATE-CLUSTERED bootstrap (resample the question
TEMPLATES with replacement, pool all of the drawn templates' questions, 2,000
reps, seed 12345; each question's score averaged over seeds first).

Why clusters (changed 2026-10-04, round-2 review): the questions are not
independent. The hard set's 14 questions come from 7 templates
(3+2+3+2+2+1+1, see agent_eval.build_hard_questions) and the original 30 from
11 (8+5+5+1+1+3+3+1+1+1+1, see agent_eval.build_questions). Same-template
siblings share question type, required tool pattern and failure mode, so
resampling individual questions treats correlated items as independent and
understates the interval. The template is the resampling unit instead; the
earlier question-level numbers are in RESULTS.md change log 11.

Caveat kept in RESULTS.md: a percentile bootstrap of the mean of 14 (or 30)
near-binary scores undercovers near the ceiling regardless of the cluster unit.
The single earlier temperature-0 run used an exact Clopper-Pearson interval.

Answers are RE-GRADED here from the raw final answers with the current
grader (agent_eval.grade_any / tool_correct); the stored per-record flags in
the raw JSONs are left untouched (they reflect the grader at run time).
records_regraded counts how many flags changed.

Usage: python -m eval_sop.agent_summary
"""
import glob
import json
import os
import re

import numpy as np

from eval_sop.agent_eval import HARD_QPATH, QPATH, grade_any, tool_correct
from eval_sop.common import RESULTS

# Template (cluster) sizes in the order the generators emit the questions.
# hard14: agent_eval.build_hard_questions -> H1..H14.
# orig30: agent_eval.build_questions -> ids 1..30 (the 4 knowledge-base items
# are hand-written one-offs, so each is its own cluster).
TEMPLATE_SIZES = {
    "hard14": [3, 2, 3, 2, 2, 1, 1],
    "orig30": [8, 5, 5, 1, 1, 3, 3, 1, 1, 1, 1],
}


def template_index(qset, n_questions):
    """Cluster id per question position, e.g. [0,0,0,1,1,2,2,2,...]."""
    sizes = TEMPLATE_SIZES[qset]
    assert sum(sizes) == n_questions, (qset, sizes, n_questions)
    return np.repeat(np.arange(len(sizes)), sizes)


def cluster_boot_ci(per_q, clusters, rng):
    """Percentile CI from resampling CLUSTERS with replacement (2,000 reps).
    Each draw pools every question of the drawn templates and takes the mean,
    so unequal template sizes weight the draw exactly as they do the point
    estimate."""
    members = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    draws = rng.integers(0, len(members), size=(2000, len(members)))
    bs = np.array([per_q[np.concatenate([members[j] for j in row])].mean() for row in draws])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    rows = []
    groups = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "agent_eval_*_t07_s*.json"))):
        m = re.search(r"agent_eval_(.+)_(orig30|hard14)_t07_s(\d)\.json", os.path.basename(f))
        d = json.load(open(f))
        groups.setdefault((d["summary"]["model"], m.group(2)), []).append(d)
    qbank = {str(q["id"]): q for path in (QPATH, HARD_QPATH) for q in json.load(open(path))}
    changed = 0
    for runs in groups.values():
        for run in runs:
            for rec in run["records"]:
                q = qbank[str(rec["id"])]
                a = bool(grade_any(q, rec["final_answer"]))
                t = bool(tool_correct(q, rec["tool_calls"]))
                changed += (a != rec["answer_correct"]) + (t != rec["tool_correct"])
                rec["answer_correct"], rec["tool_correct"] = a, t
    print("records_regraded (flags changed vs stored):", changed)
    rng = np.random.default_rng(12345)       # question-level draws (reported for comparison)
    rng_c = np.random.default_rng(12345)     # template-cluster draws (the headline CI)
    out = []
    for (model, qset), runs in sorted(groups.items()):
        ids = [r["id"] for r in runs[0]["records"]]
        mat = {k: np.array([[float(bool(rec[k])) for rec in run["records"]] for run in runs])
               for k in ("tool_correct", "answer_correct")}
        mat["joint"] = mat["tool_correct"] * mat["answer_correct"]
        res = dict(model=model, question_set=qset, n_questions=len(ids), n_seeds=len(runs),
                   temperature=runs[0]["summary"]["temperature"], num_ctx=runs[0]["summary"]["num_ctx"],
                   forecasts=runs[0]["summary"]["forecasts"])
        clusters = template_index(qset, len(ids))
        res["n_templates"] = int(len(np.unique(clusters)))
        res["template_sizes"] = TEMPLATE_SIZES[qset]
        idx = rng.integers(0, len(ids), size=(2000, len(ids)))
        for k, name in (("tool_correct", "tool"), ("answer_correct", "answer"), ("joint", "joint")):
            per_seed = mat[k].mean(axis=1)
            per_q = mat[k].mean(axis=0)
            bs = per_q[idx].mean(axis=1)
            res[name] = dict(mean=float(per_seed.mean()), seed_std=float(per_seed.std(ddof=1)),
                             per_seed=[float(x) for x in per_seed],
                             # headline: template is the resampling unit
                             ci95=cluster_boot_ci(per_q, clusters, rng_c),
                             ci95_unit="template cluster (2,000 reps, seed 12345)",
                             # kept for comparison with the pre-2026-10-04 numbers
                             ci95_question_level=[float(np.percentile(bs, 2.5)),
                                                  float(np.percentile(bs, 97.5))])
        res["per_question_joint_over_seeds"] = dict(zip(ids, [float(x) for x in mat["joint"].mean(axis=0)]))
        out.append(res)
        print(f"{model:12s} {qset:7s} n={len(ids)}x{len(runs)} tmpl={res['n_templates']}  "
              + "  ".join(f"{n}={res[n]['mean']:.3f}±{res[n]['seed_std']:.3f} "
                          f"clust[{res[n]['ci95'][0]:.2f},{res[n]['ci95'][1]:.2f}] "
                          f"q[{res[n]['ci95_question_level'][0]:.2f},{res[n]['ci95_question_level'][1]:.2f}]"
                          for n in ("tool", "answer", "joint")))
    with open(os.path.join(RESULTS, "agent_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
