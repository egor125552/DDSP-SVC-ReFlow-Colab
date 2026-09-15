#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DDSP-SVC

python - <<'PY'
import torch, torchaudio, librosa, soundfile
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda_available", torch.cuda.is_available())
PY

python /workspace/tools/make_smoke_data.py
python /workspace/tools/make_smoke_config.py

echo "Проверяю синтаксис основных файлов..."
python -m py_compile preprocess.py train_reflow.py main_reflow.py gui_reflow.py

echo "Запускаю preprocessing на крошечном датасете..."
python preprocess.py -c configs/reflow-smoke.yaml -j 1

echo "Запускаю ровно короткий smoke training..."
python train_reflow.py -c configs/reflow-smoke.yaml
python main_reflow.py -i data/val/audio/smoke_0.wav -m exp/reflow-smoke/model_2.pt -o /tmp/smoke_output.wav -k 0 -id 1 -step 1 -method euler -ts 0.0 -pe parselmouth
test -s /tmp/smoke_output.wav
python - <<'PY'
import soundfile as sf
x, sr = sf.read("/tmp/smoke_output.wav")
print("SMOKE_INFERENCE_OK", sr, len(x))
PY

echo "Smoke test завершён."
