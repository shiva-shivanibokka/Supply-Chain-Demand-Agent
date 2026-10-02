"""Regression test for forecasting/export_forecasts.py.

Runs the export's real main() end to end, but with a stub model, so no trained
checkpoint is needed:
  * the stub's predict(loader) records the decoder_time_idx the export builds,
    and returns ordered non-negative quantiles so the sanity gate passes;
  * two dummy checkpoint files mimic forecasting/saved_model on this machine
    (epoch=00 val_loss=4.6037, epoch=01 val_loss=4.5668).

Asserts:
  1. the decoder starts at last_observed_time_idx + 1 (a FUTURE forecast);
  2. the checkpoint with the lowest val_loss is loaded.

Usage (exit code 0 = pass, 1 = fail):
  python -m eval_sop.test_export_forecasts                       # current file
  python -m eval_sop.test_export_forecasts path/to/export_forecasts.py   # e.g. original
"""
import importlib.util
import os
import sys
import tempfile

import torch

from eval_sop.common import ROOT

os.environ.setdefault("RETRAIN_SAMPLE_PARTS", "3")  # small + fast; honoured by load_and_prepare


def load_module(path):
    spec = importlib.util.spec_from_file_location("export_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "forecasting", "export_forecasts.py")
    os.chdir(ROOT)  # export uses repo-relative paths
    mod = load_module(path)
    tmp = tempfile.mkdtemp()
    for name in ["tft-best-epoch=00-val_loss=4.6037.ckpt", "tft-best-epoch=01-val_loss=4.5668.ckpt"]:
        open(os.path.join(tmp, name), "w").close()
    mod.CKPT_DIR = tmp
    mod.OUT = os.path.join(tmp, "forecasts.json")
    os.environ["EXPORT_OUT"] = mod.OUT
    seen = {"decoder": [], "ckpt": None}

    class Stub:
        def eval(self):
            return self

        def predict(self, loader, mode="quantiles", return_y=False):
            x, _ = next(iter(loader))
            d = x["decoder_time_idx"][0]
            seen["decoder"].append((int(d[0]), int(d[-1])))
            base = d.float().view(1, -1, 1)
            return torch.cat([base, base + 1, base + 2], dim=2)

    import pytorch_forecasting

    def fake_load(p, *a, **k):
        seen["ckpt"] = os.path.basename(p)
        return Stub()

    pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint = staticmethod(fake_load)
    mod.main()

    from forecasting.model import load_and_prepare
    last = int(load_and_prepare(mod.DATA)["time_idx"].max())
    first_dec = {a for a, _ in seen["decoder"]}
    ok_future = first_dec == {last + 1}
    ok_ckpt = seen["ckpt"] == "tft-best-epoch=01-val_loss=4.5668.ckpt"
    print(f"file under test : {path}")
    print(f"last observed time_idx = {last}; decoder windows seen = {sorted(set(seen['decoder']))}")
    print(f"[{'PASS' if ok_future else 'FAIL'}] decoder starts at last_observed+1")
    print(f"[{'PASS' if ok_ckpt else 'FAIL'}] best checkpoint loaded (got {seen['ckpt']})")
    sys.exit(0 if (ok_future and ok_ckpt) else 1)


if __name__ == "__main__":
    main()
