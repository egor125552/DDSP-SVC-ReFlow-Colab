# DDSP-SVC ReFlow Colab

A reproducible Google Colab workflow for DDSP-SVC ReFlow with dataset preparation, preprocessing, training, checkpoint resume, inference, experimental browser realtime, Google Drive persistence, validation tooling, and Docker smoke tests.

The project is designed around one practical problem: Colab sessions are temporary, GPU environments change, datasets can be broken, and a failed training run should not silently corrupt the state you want to resume later.

## What is included

- Russian Gradio interface;
- ZIP dataset import;
- WAV normalization to 44.1 kHz mono;
- deterministic train/validation split;
- preprocessing with ContentVec and F0 extraction;
- configurable training;
- checkpoint discovery and resume;
- WAV inference;
- experimental browser realtime through Gradio;
- Google Drive persistence for datasets, extracted features, and checkpoints;
- fast local training cache;
- dataset quality analysis;
- Dataset ID tracking;
- preprocessing integrity validation;
- checkpoint-to-dataset binding;
- system grader for environment and training readiness;
- CPU Docker smoke test;
- validation against the official Google Colab runtime image.

## Open in Google Colab

https://colab.research.google.com/github/egor125552/DDSP-SVC-ReFlow-Colab/blob/main/DDSP_SVC_ReFlow_Colab.ipynb

Use a GPU runtime for real training.

The interface starts conservatively on Tesla T4. If VRAM allows it, batch size can be increased. If CUDA runs out of memory, reduce the batch size.

Before the UI starts, the notebook checks CUDA availability, GPU model, VRAM, a short FP16 computation, and RMVPE loading. If Colab provides only CPU, the workflow stops with a clear message instead of accidentally starting extremely slow training.

## Reproducibility

The upstream DDSP-SVC source is pinned to a known commit:

`3635301027473c6662d05a1c73ef34fba7f15f90`

The Colab notebook and Docker smoke test use the same upstream revision.

This prevents a future upstream change from silently breaking preprocessing or checkpoint compatibility.

Updating the upstream revision should be treated as an explicit change followed by a complete smoke test.

## Verified Colab environment

A full smoke test was run on September 16, 2026 inside the official Google Colab runtime image:

`us-docker.pkg.dev/colab-images/public/runtime:latest`

Verified image digest:

`sha256:c4375de125f45948a10009001df52774da2573ea7bb2903f8bf945ec72690c5a`

The tested environment included:

- Python 3.12.13;
- PyTorch 2.11.0+cu128;
- CUDA runtime 12.8.

The server used for this verification did not have a physical NVIDIA GPU, so CUDA execution on an actual T4 was not covered by that specific test.

Inside the official Colab image, the following workflow was exercised through the project UI:

- ZIP import;
- train/validation split;
- WAV normalization;
- preprocessing;
- two ReFlow training steps;
- validation;
- checkpoint creation;
- checkpoint discovery;
- standard WAV inference;
- direct realtime inference path.

CPU preprocessing and inference use Parselmouth automatically. CUDA mode keeps RMVPE.

## Persistent datasets and checkpoints

Colab sessions are disposable, so durable state is stored in Google Drive.

The workflow supports:

- `/content/DDSP-SVC/exp` backed by Google Drive;
- `/content/DDSP-SVC/data` backed by Google Drive;
- persistent WAV files;
- persistent F0, mel, units, volume, augmented features, and pitch augmentation metadata;
- persistent model checkpoints;
- resume after a new Colab session.

A tested checkpoint, `model_2.pt`, was successfully written to the Drive-backed experiment directory.

## Fast local training cache

Google Drive is useful for persistence but slower than Colab local storage.

The UI therefore provides a fast local cache option.

Before training, prepared features can be copied from persistent storage to `/content/ddsp-local-data`. Training then reads from the local Colab disk while checkpoints continue to be written to Google Drive.

Before the cache is created, free local space is checked. If there is not enough room, the UI blocks the copy instead of filling the runtime disk.

## Atomic dataset import

Dataset replacement is atomic.

A new ZIP is first unpacked and validated in a staging directory. Persistent train/validation data is replaced only after the new dataset passes preparation.

This protects an existing working dataset from:

- corrupted ZIP archives;
- unsafe paths such as `../`;
- archives with too few usable files;
- individual broken WAV files;
- suspicious compression ratios;
- oversized archives;
- insufficient local disk space.

Current limits include up to 20,000 files and up to 20 GB of extracted data, while keeping approximately 2 GB of local reserve.

Audio shorter than 0.5 seconds is skipped. Exact duplicates are removed. Train/validation shuffling is reproducible.

## Dataset quality analysis

The Dataset tab can analyze a reproducible sample of up to 500 files.

For each file it measures:

- duration;
- peak level;
- RMS;
- near-silence ratio;
- digital clipping ratio.

Suspicious audio is surfaced as a warning or problem depending on how much of the dataset is affected.

The report is cached by Dataset ID.

## Dataset ID

Every prepared dataset receives an ID derived from the normalized WAV contents.

The experiment remembers that ID.

If a checkpoint belongs to another dataset or has unknown provenance, training is blocked by default instead of silently mixing voices or states.

Intentional fine-tuning on another dataset requires an explicit option.

## Preprocessing integrity

Training is not allowed to start merely because a preprocessing directory exists.

For each WAV, the workflow checks the required artifacts, including:

- F0;
- volume;
- augmented volume;
- mel;
- augmented mel;
- units;
- pitch augmentation metadata.

Missing, empty, or damaged features block training and surface concrete errors.

After successful preprocessing, `data/preprocessing_manifest.json` stores the Dataset ID and important extraction parameters.

If a new Colab session finds matching intact features and a compatible manifest, preprocessing does not need to run again.

## Checkpoint provenance

Dataset identity is bound not only to the experiment directory but also to individual `model_*.pt` checkpoints through sidecar metadata.

This matters when a run crashes during fine-tuning.

An older checkpoint should not suddenly inherit the identity of a new dataset just because the active experiment changed.

The workflow keeps pending training metadata, associates newly created checkpoints with the correct Dataset ID, and cleans orphaned sidecars after training.

Legacy checkpoints without provenance metadata can be migrated explicitly through the UI after validation.

## Resume behavior

Training automatically discovers the checkpoint with the highest step number in the selected experiment.

The upstream trainer restores model weights and global step.

Optimizer state is saved by default for more complete resume behavior. This increases checkpoint size and can be disabled when storage matters more than optimizer continuity.

## Adaptive segment length

Training segment duration is not hard-coded to two seconds.

It is selected from the shortest accepted clip, with a maximum of two seconds.

This protects the upstream DataLoader from pathological behavior on short datasets.

## Environment grader

The UI includes a dedicated grader.

It checks:

- upstream source;
- CUDA/GPU availability;
- required pretrained models;
- train/validation data;
- Dataset ID;
- preprocessing integrity;
- preprocessing manifest compatibility;
- local disk reserve;
- latest checkpoint;
- optimizer state;
- checkpoint Dataset ID.

The result is shown as explicit OK, warning, or problem states rather than an arbitrary numeric score.

## Realtime

The original DDSP-SVC realtime path uses local audio devices through `sounddevice`.

That model does not map directly to Colab because the microphone lives in the browser while Python runs on a remote machine.

This project therefore includes an experimental browser realtime path through Gradio.

It keeps previous audio context and sends streaming chunks to the model.

This is not expected to match the latency of a local native realtime application.

The realtime handler has been tested on Linux and produces a valid 44.1 kHz stream. Real browser microphone latency on a Tesla T4 should still be measured inside an active Colab GPU session.

## Linux and Docker smoke test

A separate CPU Docker smoke test verifies the workflow without requiring a GPU.

It covers dependency installation, dataset preparation, preprocessing, a short training run, checkpoint save/reload, WAV inference, and the direct realtime handler.

Build:

```bash
docker build -f Dockerfile.cpu-smoke -t ddsp-svc-reflow-smoke .
```

Download pretrained files:

```bash
docker run --rm -v ddsp-pretrain:/workspace/DDSP-SVC/pretrain \
  ddsp-svc-reflow-smoke \
  bash /workspace/tools/download_pretrained.sh
```

## Known environment caveat

The upstream DDSP-SVC requirements currently downgrade NumPy to 1.26.4.

That may produce dependency warnings for unrelated packages already present in the Colab image.

The tested DDSP-SVC and Gradio workflow continued to function in the verified environment.

## Documentation

See `docs/overview.md` for a concise technical overview intended for external readers.

## Repository

https://github.com/egor125552/DDSP-SVC-ReFlow-Colab
