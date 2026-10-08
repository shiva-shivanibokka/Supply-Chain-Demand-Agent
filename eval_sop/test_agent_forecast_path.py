"""Regression test for the Streamlit/Python agent forecast path
(agent/agent.py: get_demand_forecast -> _forecast_with_tft).

Same idea as test_export_forecasts.py: a stub model records the decoder
time_idx it is asked to predict, two dummy checkpoints mimic the local
forecasting/saved_model, and MLflow logging is stubbed out (nothing written).

Asserts: (1) the TFT path is used, (2) the decoder starts at
last_observed + 1, (3) the lowest-val_loss checkpoint is loaded.

Usage: python -m eval_sop.test_agent_forecast_path [path/to/agent.py]
"""
import importlib.util
import os
import sys
import tempfile

import torch

from eval_sop.common import ROOT

os.environ.setdefault("RETRAIN_SAMPLE_PARTS", "3")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "agent", "agent.py")
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._log_forecast_to_mlflow = lambda *a, **k: None  # no MLflow / prediction-log writes

    tmp = tempfile.mkdtemp()
    for name in ["tft-best-epoch=00-val_loss=4.6037.ckpt", "tft-best-epoch=01-val_loss=4.5668.ckpt"]:
        open(os.path.join(tmp, name), "w").close()
    seen = {"decoder": None, "ckpt": None}

    class Stub:
        def eval(self):
            return self

        def predict(self, loader, mode="quantiles", return_y=False):
            x, _ = next(iter(loader))
            d = x["decoder_time_idx"][0]
            seen["decoder"] = (int(d[0]), int(d[-1]))
            base = torch.ones(1, d.shape[0], 1)
            return torch.cat([base, base * 2, base * 3], dim=2)

    import pytorch_forecasting

    def fake_load(p, *a, **k):
        seen["ckpt"] = os.path.basename(p)
        return Stub()

    pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint = staticmethod(fake_load)
    out = mod.get_demand_forecast("PART_001", model_dir=tmp)

    from forecasting.model import load_and_prepare
    last = int(load_and_prepare("data/supply_chain_data.csv")["time_idx"].max())
    ok_tft = "TFT model" in out
    ok_future = seen["decoder"] is not None and seen["decoder"][0] == last + 1
    ok_ckpt = seen["ckpt"] == "tft-best-epoch=01-val_loss=4.5668.ckpt"
    print(f"file under test : {os.path.relpath(path, ROOT)}")
    print(f"last observed time_idx = {last}; decoder = {seen['decoder']}")
    print(f"[{'PASS' if ok_tft else 'FAIL'}] TFT path used")
    print(f"[{'PASS' if ok_future else 'FAIL'}] decoder starts at last_observed+1")
    print(f"[{'PASS' if ok_ckpt else 'FAIL'}] best checkpoint loaded (got {seen['ckpt']})")
    sys.exit(0 if (ok_tft and ok_future and ok_ckpt) else 1)


if __name__ == "__main__":
    main()
