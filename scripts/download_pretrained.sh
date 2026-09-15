#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/workspace/DDSP-SVC}"
cd "$ROOT"

mkdir -p pretrain/contentvec pretrain/nsf_hifigan pretrain
echo "Скачиваю ContentVec..."
curl -L --fail --retry 3   "https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin?download=true"   -o pretrain/contentvec/pytorch_model.bin

echo "Скачиваю NSF-HiFiGAN..."
tmpdir="$(mktemp -d)"
curl -L --fail --retry 3   "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"   -o "$tmpdir/vocoder.zip"
unzip -q "$tmpdir/vocoder.zip" -d "$tmpdir/vocoder"

model_file="$(find "$tmpdir/vocoder" -type f \( -name model -o -name '*.ckpt' -o -name '*.pt' \) | head -n 1)"
config_file="$(find "$tmpdir/vocoder" -type f -name config.json | head -n 1)"
test -n "$model_file"
test -n "$config_file"
cp "$model_file" pretrain/nsf_hifigan/model
cp "$config_file" pretrain/nsf_hifigan/config.json
rm -rf "$tmpdir"

echo "Скачиваю RMVPE..."
tmpdir="$(mktemp -d)"
curl -L --fail --retry 3   "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"   -o "$tmpdir/rmvpe.zip"
unzip -q "$tmpdir/rmvpe.zip" -d "$tmpdir/rmvpe"
mkdir -p pretrain/rmvpe
rmvpe_model="$(find "$tmpdir/rmvpe" -type f -name model.pt | head -n 1)"
test -n "$rmvpe_model"
cp "$rmvpe_model" pretrain/rmvpe/model.pt
rm -rf "$tmpdir"

echo "Готово. Предобученные файлы находятся в $ROOT/pretrain"
