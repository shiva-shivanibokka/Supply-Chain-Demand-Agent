"""Unit test for forecasting.export_forecasts._best_checkpoint (no model needed).

Lightning appends '-v1', '-v2', ... on filename clashes; such checkpoints must
still be ranked by their val_loss.

Usage: python -m eval_sop.test_best_checkpoint   (exit 0 = pass)
"""
import sys

from forecasting.export_forecasts import _best_checkpoint

CASES = [
    (["d/tft-best-epoch=00-val_loss=4.6037.ckpt", "d/tft-best-epoch=01-val_loss=4.5668.ckpt"],
     "d/tft-best-epoch=01-val_loss=4.5668.ckpt", "plain names"),
    (["d/tft-best-epoch=00-val_loss=4.6037.ckpt", "d/tft-best-epoch=03-val_loss=4.5000-v1.ckpt"],
     "d/tft-best-epoch=03-val_loss=4.5000-v1.ckpt", "versioned name (-v1) is best"),
    (["d/tft-best-epoch=02-val_loss=4.4000-v2.ckpt", "d/tft-best-epoch=03-val_loss=4.5000-v1.ckpt"],
     "d/tft-best-epoch=02-val_loss=4.4000-v2.ckpt", "two versioned names"),
]


def main():
    fails = 0
    for ckpts, exp, desc in CASES:
        got = _best_checkpoint(ckpts)
        ok = got == exp
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}: expected {exp.split('/')[-1]}, got {got.split('/')[-1]}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
