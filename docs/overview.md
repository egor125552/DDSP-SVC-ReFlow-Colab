# DDSP-SVC ReFlow Colab technical overview

## Purpose

DDSP-SVC ReFlow Colab turns an upstream voice-conversion project into a reproducible training workflow that can survive the practical problems of Google Colab.

The main engineering work is not merely placing commands into a notebook.

The project adds state management, validation, dataset provenance, persistence, environment checks, recovery behavior, and reproducible testing around the upstream model.

## Main workflow

A user can:

- open the notebook in Google Colab;
- mount Google Drive;
- import a voice dataset ZIP;
- normalize and split the audio;
- run preprocessing;
- inspect dataset quality;
- start or resume training;
- save checkpoints persistently;
- run WAV inference;
- test an experimental browser realtime path.

## Why persistence matters

A Colab runtime can disappear at any time.

If training data, extracted features, or checkpoints exist only on the temporary runtime disk, hours of work may be lost.

This project therefore separates persistent state from fast local working state.

Google Drive holds durable datasets and checkpoints. Optional local caching accelerates training while preserving durable outputs.

## Dataset safety

Dataset import is staged and validated before replacing existing persistent data.

The importer checks archive structure, path safety, file count, extracted size, compression behavior, free disk space, damaged WAV files, clip duration, and duplicates.

This means a broken upload should fail before it destroys a previously working dataset.

## Dataset identity

A Dataset ID is derived from normalized source audio.

That identity is stored alongside preprocessing state and checkpoints.

If the selected checkpoint does not match the current dataset, training is blocked unless the user explicitly requests cross-dataset fine-tuning.

This protects against a subtle but serious failure mode: accidentally resuming a model with data from another voice and assuming the run is still consistent.

## Preprocessing validation

The workflow validates that each source WAV has the feature files required by training.

It also writes a preprocessing manifest containing dataset identity and extraction parameters.

A later session can safely reuse preprocessing only when the files are intact and the manifest still matches the active configuration.

## Checkpoint provenance

Checkpoint identity is tracked per model file.

A crash during fine-tuning should not retroactively relabel an older checkpoint as belonging to the new dataset.

Pending training metadata and checkpoint sidecars preserve the relationship between model files and the dataset that actually produced them.

## Reproducibility

The upstream DDSP-SVC source is pinned to a specific commit.

The workflow has been smoke-tested inside the official Google Colab runtime image and separately in a CPU Docker environment.

The goal is to make failures explainable.

If the environment changes, the correct response is to rerun compatibility checks rather than silently assuming that a notebook that worked last week must still work today.

## Realtime architecture

Local DDSP-SVC realtime uses physical audio devices on the same machine.

Colab runs remotely, so browser microphone audio must cross a browser-to-server boundary.

The project therefore provides a separate experimental Gradio streaming path.

It maintains audio context between chunks and performs model inference remotely.

This design is functional, but it is intentionally not advertised as equal to native local realtime latency.

## System grader

The project includes a grader that checks whether the environment is ready to train.

It verifies the source tree, GPU state, pretrained files, dataset identity, preprocessing integrity, manifest compatibility, disk space, checkpoint state, optimizer state, and checkpoint provenance.

The grader reports concrete states instead of hiding multiple failure modes behind one synthetic score.

## Technology

The workflow uses:

- Python;
- PyTorch;
- DDSP-SVC ReFlow;
- Gradio;
- Google Colab;
- CUDA;
- ContentVec;
- RMVPE;
- Parselmouth;
- Google Drive;
- Docker;
- Jupyter notebooks.

## Testing status

Verified paths include:

- dataset import;
- normalization;
- deterministic split;
- preprocessing;
- short training;
- validation;
- checkpoint creation;
- checkpoint rediscovery;
- persistent Google Drive storage;
- local training cache;
- WAV inference;
- direct realtime handler;
- CPU Docker smoke test.

A real NVIDIA T4 execution path should still be rechecked inside a live GPU-backed Colab session whenever the runtime image or upstream model revision changes.

## Links

GitHub:

https://github.com/egor125552/DDSP-SVC-ReFlow-Colab

Colab:

https://colab.research.google.com/github/egor125552/DDSP-SVC-ReFlow-Colab/blob/main/DDSP_SVC_ReFlow_Colab.ipynb
