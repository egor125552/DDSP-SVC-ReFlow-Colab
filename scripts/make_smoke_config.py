from pathlib import Path
import yaml

root = Path("/workspace/DDSP-SVC")
src = root / "configs" / "reflow.yaml"
dst = root / "configs" / "reflow-smoke.yaml"
cfg = yaml.safe_load(src.read_text())
cfg["device"] = "cpu"
cfg["data"]["f0_extractor"] = "parselmouth"
cfg["data"]["duration"] = 0.5
cfg["model"]["n_aux_layers"] = 1
cfg["model"]["n_aux_chans"] = 64
cfg["model"]["n_layers"] = 1
cfg["model"]["n_chans"] = 128
cfg["model"]["use_attention"] = False
cfg["train"]["num_workers"] = 0
cfg["train"]["amp_dtype"] = "fp32"
cfg["train"]["batch_size"] = 1
cfg["train"]["cache_all_data"] = False
cfg["train"]["epochs"] = 1
cfg["train"]["interval_log"] = 1
cfg["train"]["interval_val"] = 2
cfg["train"]["interval_force_save"] = 2
cfg["infer"]["infer_step"] = 1
cfg["env"]["expdir"] = "exp/reflow-smoke"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(dst)
