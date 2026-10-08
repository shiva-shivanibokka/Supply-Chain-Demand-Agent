"""Unit test for forecasting.export_forecasts._best_checkpoint (no model needed).

Lightning appends '-v1', '-v2', ... on filename clashes; such checkpoints must
still be ranked by their val_loss. An INTEGER val_loss ("val_loss=4.ckpt",
"val_loss=4") must also be ranked, and a name with no parseable val_loss must
RAISE instead of being silently scored as inf.

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
    # ADDED 2026-10-04 (round-2 review): an integer val_loss used to score inf.
    (["d/tft-best-epoch=00-val_loss=4.6037.ckpt", "d/tft-best-epoch=05-val_loss=4.ckpt"],
     "d/tft-best-epoch=05-val_loss=4.ckpt", "integer val_loss with .ckpt suffix"),
    (["d/tft-best-epoch=00-val_loss=4.6037.ckpt", "d/tft-best-epoch=05-val_loss=4"],
     "d/tft-best-epoch=05-val_loss=4", "integer val_loss, no extension"),
    (["d/tft-best-epoch=05-val_loss=5.ckpt", "d/tft-best-epoch=00-val_loss=4.6037.ckpt"],
     "d/tft-best-epoch=00-val_loss=4.6037.ckpt", "integer val_loss is not always best"),
]

# Names with no parseable val_loss must raise, not be silently ranked as inf.
RAISE_CASES = [
    (["d/tft-best-epoch=00.ckpt"], "no val_loss at all"),
    (["d/tft-best-epoch=00-val_loss=4.6037.ckpt", "d/last.ckpt"], "one unparseable name among good ones"),
]


def main():
    fails = 0
    for ckpts, exp, desc in CASES:
        got = _best_checkpoint(ckpts)
        ok = got == exp
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}: expected {exp.split('/')[-1]}, got {got.split('/')[-1]}")
    for ckpts, desc in RAISE_CASES:
        try:
            got = _best_checkpoint(ckpts)
            ok, detail = False, f"returned {got.split('/')[-1]} instead of raising"
        except ValueError as e:
            ok, detail = True, f"raised ValueError: {e}"
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}: {detail}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
