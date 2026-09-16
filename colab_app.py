import os
import sys
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

DDSP_ROOT = Path(os.environ.get("DDSP_ROOT", "/content/DDSP-SVC")).resolve()
if str(DDSP_ROOT) not in sys.path:
    sys.path.insert(0, str(DDSP_ROOT))
if DDSP_ROOT.exists():
    os.chdir(DDSP_ROOT)
TRAIN_PROCESS = None
TRAIN_LOG = None
TRAIN_EXP_NAME = None
TRAIN_FINAL_CHECK = None
_RT_MODEL = None
_RT_MODEL_PATH = None
_RT_BUFFER = np.zeros(0, dtype=np.float32)

def root_ok():
    return (DDSP_ROOT / "train_reflow.py").exists()

def training_is_running():
    return TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None

def validate_exp_name(exp_name):
    name = str(exp_name or "").strip()
    if not name:
        raise ValueError("Имя эксперимента не может быть пустым.")
    if name in {".", ".."}:
        raise ValueError("Недопустимое имя эксперимента.")
    if "/" in name or "\\" in name:
        raise ValueError("В имени эксперимента нельзя использовать / или \\.")
    if len(name) > 80:
        raise ValueError("Имя эксперимента слишком длинное. Максимум 80 символов.")
    if any(ord(ch) < 32 for ch in name):
        raise ValueError("В имени эксперимента есть управляющие символы.")
    return name

def tail_text(path, lines=80):
    p = Path(path)
    if not p.exists():
        return "Журнал пока пуст."
    data = p.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])

def human_bytes(value):
    value = float(value)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ТБ"

def directory_size(path):
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for item in root.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total

def free_space(path):
    try:
        return shutil.disk_usage(Path(path)).free
    except Exception:
        return None

def calculate_dataset_fingerprint():
    digests = []
    for folder in ("train/audio", "val/audio"):
        for path in sorted((DDSP_ROOT / "data" / folder).glob("*.wav")):
            try:
                digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                continue
    if not digests:
        return ""
    return hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()

def current_dataset_fingerprint():
    marker = DDSP_ROOT / "data/dataset_fingerprint.txt"
    if marker.exists():
        value = marker.read_text(errors="replace").strip()
        if value:
            return value
    value = calculate_dataset_fingerprint()
    if value:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(value + "\n")
    return value

def training_segment_duration():
    wavs = list((DDSP_ROOT / "data/train/audio").glob("*.wav"))
    wavs += list((DDSP_ROOT / "data/val/audio").glob("*.wav"))
    durations = []
    for path in wavs:
        try:
            info = sf.info(path)
            if info.samplerate > 0 and info.frames > 0:
                durations.append(info.frames / info.samplerate)
        except Exception:
            continue
    if not durations:
        return 2.0
    shortest = min(durations)
    return round(max(0.25, min(2.0, shortest - 0.05)), 3)

def preprocessing_integrity():
    required_features = ("f0", "volume", "aug_vol", "mel", "aug_mel", "units")
    total = 0
    complete = 0
    problems = []

    for split in ("train", "val"):
        root = DDSP_ROOT / "data" / split
        audio_dir = root / "audio"
        wavs = sorted(audio_dir.glob("*.wav"))
        total += len(wavs)

        pitch_dict = {}
        pitch_path = root / "pitch_aug_dict.npy"
        if pitch_path.exists():
            try:
                loaded = np.load(pitch_path, allow_pickle=True)
                pitch_dict = loaded.item() if getattr(loaded, "shape", None) == () else {}
                if not isinstance(pitch_dict, dict):
                    pitch_dict = {}
            except Exception as exc:
                problems.append(f"{split}/pitch_aug_dict.npy повреждён: {exc}")
        elif wavs:
            problems.append(f"{split}/pitch_aug_dict.npy отсутствует")

        for wav in wavs:
            name_ext = wav.name
            file_ok = True
            for feature in required_features:
                path = root / feature / f"{name_ext}.npy"
                if not path.exists():
                    problems.append(f"{split}/{feature}/{name_ext}.npy отсутствует")
                    file_ok = False
                    continue
                try:
                    arr = np.load(path, mmap_mode="r")
                    if getattr(arr, "size", 0) <= 0 or getattr(arr, "ndim", 0) <= 0:
                        raise ValueError("пустой массив")
                except Exception as exc:
                    problems.append(f"{split}/{feature}/{name_ext}.npy повреждён: {exc}")
                    file_ok = False

            if name_ext not in pitch_dict:
                problems.append(f"{split}/pitch_aug_dict.npy не содержит {name_ext}")
                file_ok = False

            if file_ok:
                complete += 1

    ready = total > 0 and complete == total and not problems
    return {
        "ready": ready,
        "total": total,
        "complete": complete,
        "problems": problems,
    }

def preprocessing_manifest_for_config(cfg):
    data = cfg.get("data", {})
    return {
        "dataset_fingerprint": current_dataset_fingerprint(),
        "f0_extractor": data.get("f0_extractor"),
        "sampling_rate": data.get("sampling_rate"),
        "block_size": data.get("block_size"),
        "encoder": data.get("encoder"),
        "encoder_sample_rate": data.get("encoder_sample_rate"),
        "encoder_hop_size": data.get("encoder_hop_size"),
        "encoder_out_channels": data.get("encoder_out_channels"),
    }

def read_preprocessing_manifest():
    path = DDSP_ROOT / "data/preprocessing_manifest.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(errors="replace"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None

def make_config(batch_size=32, cache_all=False, exp_name="reflow-colab", epochs=100000, interval_val=2000, interval_force_save=10000, fast_local_data=False, save_optimizer=True):
    exp_name = validate_exp_name(exp_name)
    src = DDSP_ROOT / "configs" / "reflow.yaml"
    dst = DDSP_ROOT / "configs" / "reflow-colab.yaml"
    cfg = yaml.safe_load(src.read_text())
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["data"]["f0_extractor"] = "rmvpe" if torch.cuda.is_available() else "parselmouth"
    cfg["data"]["duration"] = training_segment_duration()
    if fast_local_data:
        local_data = Path(os.environ.get("DDSP_LOCAL_DATA", "/content/ddsp-local-data")).resolve()
        cfg["data"]["train_path"] = str(local_data / "train")
        cfg["data"]["valid_path"] = str(local_data / "val")
    cfg["env"]["expdir"] = f"exp/{exp_name}"
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["cache_all_data"] = bool(cache_all)
    cfg["train"]["cache_device"] = "cpu"
    cfg["train"]["amp_dtype"] = "fp16" if torch.cuda.is_available() else "fp32"
    cfg["train"]["num_workers"] = 2 if torch.cuda.is_available() else 0
    cfg["train"]["epochs"] = max(1, int(epochs))
    cfg["train"]["interval_val"] = max(1, int(interval_val))
    cfg["train"]["interval_force_save"] = max(int(interval_force_save), int(interval_val))
    cfg["train"]["save_opt"] = bool(save_optimizer)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return dst

def system_status():
    gpu = "CUDA недоступна"
    if torch.cuda.is_available():
        gpu = f"{torch.cuda.get_device_name(0)}, VRAM {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} ГБ"
    train_wavs = len(list((DDSP_ROOT / "data/train/audio").glob("*.wav")))
    val_wavs = len(list((DDSP_ROOT / "data/val/audio").glob("*.wav")))
    dataset_fp = current_dataset_fingerprint()
    integrity = preprocessing_integrity()
    manifest = read_preprocessing_manifest()
    manifest_dataset = manifest.get("dataset_fingerprint", "") if manifest else ""
    manifest_matches_dataset = bool(dataset_fp and manifest_dataset == dataset_fp)
    exp_root = DDSP_ROOT / "exp"
    checkpoints = list(exp_root.glob("*/model_*.pt")) if exp_root.exists() else []
    last_ckpt = max(checkpoints, key=_checkpoint_step) if checkpoints else None

    local_root = Path("/content") if Path("/content").exists() else DDSP_ROOT
    local_free = free_space(local_root)
    drive_root = Path("/content/drive/MyDrive/DDSP-SVC-ReFlow")
    drive_free = free_space(drive_root) if drive_root.exists() else None
    data_bytes = directory_size(DDSP_ROOT / "data")

    parts = [
        f"Папка DDSP-SVC: {DDSP_ROOT}",
        f"Исходники: {'найдены' if root_ok() else 'не найдены'}",
        f"PyTorch: {torch.__version__}",
        f"GPU: {gpu}",
        f"Датасет: train {train_wavs} WAV, val {val_wavs} WAV",
        f"Датасет ID: {dataset_fp[:12] if dataset_fp else 'нет'}",
        (
            f"Preprocessing: готов, {integrity['complete']}/{integrity['total']} файлов"
            if integrity["ready"] and manifest_matches_dataset
            else (
                f"Preprocessing: файлы целы, но manifest не совпадает с текущим датасетом"
                if integrity["ready"]
                else f"Preprocessing: не готов, {integrity['complete']}/{integrity['total']} файлов полностью подготовлено"
            )
        ),
        f"Сегмент обучения: {training_segment_duration():.3f} с",
        f"Последний checkpoint: {last_ckpt.name if last_ckpt else 'нет'}",
        f"Размер persistent data: {human_bytes(data_bytes)}",
        f"Свободно локально: {human_bytes(local_free) if local_free is not None else 'не удалось определить'}",
        f"Свободно на Google Drive: {human_bytes(drive_free) if drive_free is not None else 'не удалось определить'}",
    ]
    for p in [
        DDSP_ROOT / "pretrain/contentvec/pytorch_model.bin",
        DDSP_ROOT / "pretrain/nsf_hifigan/model",
        DDSP_ROOT / "pretrain/nsf_hifigan/config.json",
        DDSP_ROOT / "pretrain/rmvpe/model.pt",
    ]:
        parts.append(f"{p.name}: {'есть' if p.exists() else 'нет'}")
    return "\n".join(parts)

def prepare_dataset(zip_path, validation_count):
    if training_is_running():
        return "Сначала останови обучение. Нельзя менять датасет, пока trainer читает его файлы."
    if not zip_path:
        return "Нужен ZIP с WAV файлами."

    data_root = DDSP_ROOT / "data"
    train_root = data_root / "train"
    val_root = data_root / "val"
    tmp = Path(tempfile.mkdtemp(prefix="ddsp_dataset_"))
    stage = Path(tempfile.mkdtemp(prefix="ddsp_stage_"))

    try:
        try:
            with zipfile.ZipFile(zip_path) as z:
                for member in z.infolist():
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(f"Небезопасный путь в ZIP: {member.filename}")
                z.extractall(tmp)
        except (zipfile.BadZipFile, ValueError) as exc:
            return f"ZIP не принят: {exc}. Старый датасет не изменён."

        wavs = sorted([p for p in tmp.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"])
        if len(wavs) < 3:
            return "Нужно хотя бы 3 WAV файла. Старый датасет не изменён."

        prepared = []
        skipped_broken = []
        skipped_short = []
        skipped_duplicates = []
        seen_audio = set()
        for src in wavs:
            try:
                audio, _ = librosa.load(src, sr=44100, mono=True)
                if audio is None or len(audio) == 0:
                    raise ValueError("пустой аудиофайл")
                duration = len(audio) / 44100
                if duration < 0.5:
                    skipped_short.append((src.name, duration))
                    continue

                out = stage / f"{len(prepared):05d}.wav"
                sf.write(out, audio, 44100)
                digest = hashlib.sha256(out.read_bytes()).hexdigest()
                if digest in seen_audio:
                    skipped_duplicates.append(src.name)
                    out.unlink(missing_ok=True)
                    continue
                seen_audio.add(digest)
                prepared.append(out)
            except Exception as exc:
                skipped_broken.append((src.name, str(exc)))

        if len(prepared) < 3:
            return (
                f"После проверки осталось только {len(prepared)} исправных WAV из {len(wavs)}. "
                "Нужно хотя бы 3. Старый датасет не изменён."
            )

        # Фиксированное перемешивание не зависит от порядка файлов в ZIP,
        # но даёт воспроизводимый train/val split между повторными импортами.
        random.Random(42).shuffle(prepared)
        n_val = max(1, min(int(validation_count), len(prepared) // 5))
        new_train = stage / "final_train"
        new_val = stage / "final_val"
        (new_train / "audio").mkdir(parents=True, exist_ok=True)
        (new_val / "audio").mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(prepared):
            target_dir = new_val / "audio" if idx < n_val else new_train / "audio"
            shutil.copy2(src, target_dir / f"{idx:05d}.wav")

        # Commit: persistent data меняем только после успешной подготовки staging.
        shutil.rmtree(train_root, ignore_errors=True)
        shutil.rmtree(val_root, ignore_errors=True)
        shutil.copytree(new_train, train_root)
        shutil.copytree(new_val, val_root)
        dataset_fingerprint = calculate_dataset_fingerprint()
        if dataset_fingerprint:
            (data_root / "dataset_fingerprint.txt").write_text(dataset_fingerprint + "\n")
        (data_root / "preprocessing_manifest.json").unlink(missing_ok=True)

        skipped_notes = []
        if skipped_broken:
            skipped_notes.append(f"битых: {len(skipped_broken)}")
        if skipped_short:
            skipped_notes.append(f"короче 0,5 с: {len(skipped_short)}")
        if skipped_duplicates:
            skipped_notes.append(f"точных дублей: {len(skipped_duplicates)}")
        skipped_note = f" Пропущено WAV ({', '.join(skipped_notes)})." if skipped_notes else ""
        segment = training_segment_duration()
        return (
            f"Готово. Обучение: {len(prepared)-n_val} файлов. Проверка: {n_val}. "
            f"Все приведено к 44,1 кГц mono.{skipped_note} "
            f"Сегмент обучения будет {segment:.3f} с. Старые признаки очищены, "
            "теперь нажми «Подготовить признаки»."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)

def run_preprocess(batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save):
    if training_is_running():
        return "Сначала останови обучение. Preprocessing во время training заблокирован."

    try:
        cfg_path = make_config(batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save)
    except ValueError as exc:
        return str(exc)
    cfg = yaml.safe_load(cfg_path.read_text())
    expected_manifest = preprocessing_manifest_for_config(cfg)
    integrity = preprocessing_integrity()
    current_manifest = read_preprocessing_manifest()

    if integrity["ready"] and current_manifest == expected_manifest:
        return (
            f"Preprocessing уже актуален: {integrity['complete']}/{integrity['total']} файлов готовы. "
            "Повторный запуск ContentVec/F0 не нужен."
        )

    cmd = ["python", "preprocess.py", "-c", str(cfg_path), "-j", "2"]
    p = subprocess.run(cmd, cwd=DDSP_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout[-12000:]
    if p.returncode != 0:
        raise gr.Error("Preprocessing завершился с ошибкой.\n" + output[-5000:])

    integrity_after = preprocessing_integrity()
    if not integrity_after["ready"]:
        details = "; ".join(integrity_after["problems"][:3])
        raise gr.Error(
            "Preprocessing завершился, но проверка целостности не пройдена. "
            f"Готово {integrity_after['complete']}/{integrity_after['total']}. {details}"
        )

    manifest_path = DDSP_ROOT / "data/preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(expected_manifest, ensure_ascii=False, indent=2) + "\n")
    return output + "\n\nPreprocessing проверен и manifest сохранён."

def start_training(batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save, fast_local_data, save_optimizer, allow_dataset_change):
    global TRAIN_PROCESS, TRAIN_LOG, TRAIN_EXP_NAME, TRAIN_FINAL_CHECK
    if training_is_running():
        return "Обучение уже идёт."
    try:
        exp_name = validate_exp_name(exp_name)
    except ValueError as exc:
        return str(exc)

    source = DDSP_ROOT / "data"
    integrity = preprocessing_integrity()
    if not integrity["ready"]:
        details = integrity["problems"][:3]
        suffix = f" Первые проблемы: {'; '.join(details)}" if details else ""
        return (
            "Сначала нажми «Подготовить признаки». "
            f"Полностью подготовлено {integrity['complete']} из {integrity['total']} файлов."
            f"{suffix}"
        )

    preview_cfg_path = make_config(
        batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save,
        fast_local_data=False,
        save_optimizer=bool(save_optimizer),
    )
    preview_cfg = yaml.safe_load(preview_cfg_path.read_text())
    expected_manifest = preprocessing_manifest_for_config(preview_cfg)
    current_manifest = read_preprocessing_manifest()
    if current_manifest != expected_manifest:
        return (
            "Preprocessing сделан с другой конфигурацией или manifest отсутствует. "
            "Нажми «Подготовить признаки», чтобы обновить ContentVec/F0 перед training."
        )

    dataset_fp = current_dataset_fingerprint()
    if not dataset_fp:
        return "Не удалось определить fingerprint датасета. Подготовь датасет заново."

    exp_dir = DDSP_ROOT / "exp" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    finalize_pending_training(exp_name)
    resume_from = latest_checkpoint(exp_name)
    previous_fp = checkpoint_dataset_fingerprint(resume_from)

    if resume_from and previous_fp != dataset_fp and not allow_dataset_change:
        reason = (
            "у checkpoint нет сохранённого Dataset ID"
            if not previous_fp
            else f"Dataset ID checkpoint: {previous_fp[:12]}, текущий: {dataset_fp[:12]}"
        )
        return (
            "Остановлено: найден checkpoint от другого или неизвестного датасета. "
            f"{reason}. Используй другое имя эксперимента или включи "
            "«Разрешить fine-tune на другом датасете»."
        )

    start_step = _checkpoint_step(resume_from) if resume_from else -1
    pending_path = _pending_training_path(exp_name)
    pending_path.write_text(json.dumps({
        "dataset_fingerprint": dataset_fp,
        "origin_step": start_step,
        "resume_from": Path(resume_from).name if resume_from else "",
        "fine_tune": bool(resume_from and previous_fp != dataset_fp),
    }, ensure_ascii=False, indent=2) + "\n")

    cache_note = ""
    if fast_local_data:
        local_data = Path(os.environ.get("DDSP_LOCAL_DATA", "/content/ddsp-local-data")).resolve()
        local_fp_file = local_data / "dataset_fingerprint.txt"
        cached_fp = local_fp_file.read_text(errors="replace").strip() if local_fp_file.exists() else ""
        cache_ready = (
            cached_fp == dataset_fp
            and (local_data / "train/pitch_aug_dict.npy").exists()
            and (local_data / "val/pitch_aug_dict.npy").exists()
        )
        if cache_ready:
            cache_note = "Локальный кэш уже готов, повторное копирование не нужно."
        else:
            source_bytes = directory_size(source / "train") + directory_size(source / "val")
            local_parent = local_data.parent if local_data.parent.exists() else Path("/content")
            available = free_space(local_parent)
            reserve = 1024 ** 3
            required = int(source_bytes * 1.2) + reserve
            if available is not None and available < required:
                return (
                    "Недостаточно локального места для быстрого кэша. "
                    f"Нужно примерно {human_bytes(required)}, свободно {human_bytes(available)}. "
                    "Освободи место или выключи «Быстрый локальный кэш для обучения»."
                )

            shutil.rmtree(local_data, ignore_errors=True)
            local_data.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / "train", local_data / "train")
            shutil.copytree(source / "val", local_data / "val")
            local_fp_file.write_text(dataset_fp + "\n")
            cache_note = (
                f"Признаки скопированы с Drive в быстрый локальный кэш "
                f"({human_bytes(source_bytes)})."
            )

    cfg = make_config(
        batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save,
        fast_local_data=bool(fast_local_data),
        save_optimizer=bool(save_optimizer),
    )
    log_dir = DDSP_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    TRAIN_LOG = log_dir / "training.log"
    log_file = open(TRAIN_LOG, "w", buffering=1)
    TRAIN_PROCESS = subprocess.Popen(
        ["python", "train_reflow.py", "-c", str(cfg)],
        cwd=DDSP_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    TRAIN_EXP_NAME = exp_name
    TRAIN_FINAL_CHECK = None
    if resume_from:
        if previous_fp and previous_fp != dataset_fp:
            resume_note = f"Fine-tune с {Path(resume_from).name} на новом датасете."
        elif not previous_fp:
            resume_note = f"Продолжаем legacy checkpoint {Path(resume_from).name}; Dataset ID раньше не сохранялся."
        else:
            resume_note = f"Продолжаем с {Path(resume_from).name}."
    else:
        resume_note = "Готового чекпойнта нет, начинаем с нуля."
    optimizer_note = "Состояние optimizer сохраняется." if save_optimizer else "Optimizer не сохраняется."
    cache_suffix = f" {cache_note}" if cache_note else ""
    return (
        f"Обучение запущено. PID {TRAIN_PROCESS.pid}. {resume_note} {optimizer_note}"
        f"{cache_suffix} Кнопка Статус покажет последние строки."
    )

def training_status():
    global TRAIN_PROCESS, TRAIN_EXP_NAME, TRAIN_FINAL_CHECK
    state = "не запускалось"
    code = None
    if TRAIN_PROCESS is not None:
        code = TRAIN_PROCESS.poll()
        state = "идёт" if code is None else f"завершено, код {code}"

    exp_root = DDSP_ROOT / "exp"
    if exp_root.exists():
        for pending in exp_root.glob("*/pending_dataset.json"):
            exp_name = pending.parent.name
            if training_is_running():
                sync_pending_checkpoint_identity(exp_name)
                cleanup_checkpoint_metadata(exp_name)
            else:
                finalize_pending_training(exp_name)

    if code is not None and TRAIN_EXP_NAME and TRAIN_FINAL_CHECK is None:
        latest = latest_checkpoint(TRAIN_EXP_NAME)
        if code == 0 and latest:
            TRAIN_FINAL_CHECK = "Проверка последнего checkpoint:\n" + inspect_checkpoint(latest)
        elif code == 0:
            TRAIN_FINAL_CHECK = (
                "Training завершился с кодом 0, но checkpoint model_*.pt в "
                f"exp/{TRAIN_EXP_NAME} не найден."
            )
        else:
            TRAIN_FINAL_CHECK = "Training завершился с ошибкой; checkpoint автоматически не подтверждён."

    log = tail_text(TRAIN_LOG) if TRAIN_LOG else "Журнал пока пуст."
    final = f"\n\n{TRAIN_FINAL_CHECK}" if TRAIN_FINAL_CHECK else ""
    return f"Состояние: {state}\n\n{log}{final}"

def stop_training():
    global TRAIN_PROCESS, TRAIN_FINAL_CHECK
    if TRAIN_PROCESS is None or TRAIN_PROCESS.poll() is not None:
        return "Активного обучения нет."
    TRAIN_PROCESS.terminate()
    try:
        TRAIN_PROCESS.wait(timeout=10)
    except subprocess.TimeoutExpired:
        TRAIN_PROCESS.kill()

    exp_root = DDSP_ROOT / "exp"
    if exp_root.exists():
        for pending in exp_root.glob("*/pending_dataset.json"):
            finalize_pending_training(pending.parent.name)

    TRAIN_FINAL_CHECK = "Обучение остановлено пользователем; сохранённые checkpoint не удалены."
    return TRAIN_FINAL_CHECK

def _checkpoint_step(path):
    try:
        return int(Path(path).stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1

def _checkpoint_dataset_sidecar(path):
    path = Path(path)
    return path.with_name(path.name + ".dataset_fingerprint.txt")

def checkpoint_dataset_fingerprint(path):
    if not path:
        return ""
    sidecar = _checkpoint_dataset_sidecar(path)
    if not sidecar.exists():
        return ""
    try:
        return sidecar.read_text(errors="replace").strip()
    except OSError:
        return ""

def _pending_training_path(exp_name):
    exp_name = validate_exp_name(exp_name)
    return DDSP_ROOT / "exp" / exp_name / "pending_dataset.json"

def cleanup_checkpoint_metadata(exp_name):
    exp_dir = DDSP_ROOT / "exp" / exp_name
    if not exp_dir.exists():
        return 0
    removed = 0
    suffix = ".dataset_fingerprint.txt"
    for sidecar in exp_dir.glob(f"model_*.pt{suffix}"):
        checkpoint = Path(str(sidecar)[:-len(suffix)])
        if not checkpoint.exists():
            try:
                sidecar.unlink()
                removed += 1
            except OSError:
                pass
    return removed

def finalize_pending_training(exp_name):
    sync_pending_checkpoint_identity(exp_name)
    cleanup_checkpoint_metadata(exp_name)
    pending_path = _pending_training_path(exp_name)
    if pending_path.exists():
        try:
            pending_path.unlink()
            return True
        except OSError:
            return False
    return False

def sync_pending_checkpoint_identity(exp_name):
    pending_path = _pending_training_path(exp_name)
    if not pending_path.exists():
        return ""
    try:
        pending = json.loads(pending_path.read_text(errors="replace"))
        dataset_fp = str(pending.get("dataset_fingerprint", "")).strip()
        origin_step = int(pending.get("origin_step", pending.get("start_step", -1)))
    except Exception:
        return ""

    exp_dir = DDSP_ROOT / "exp" / exp_name
    checkpoints = sorted(exp_dir.glob("model_*.pt"), key=_checkpoint_step)
    if not dataset_fp or not checkpoints:
        return ""

    labeled = []
    for checkpoint in checkpoints:
        step = _checkpoint_step(checkpoint)
        if step <= origin_step:
            continue
        sidecar = _checkpoint_dataset_sidecar(checkpoint)
        existing_fp = checkpoint_dataset_fingerprint(checkpoint)
        if existing_fp and existing_fp != dataset_fp:
            continue
        if not existing_fp:
            sidecar.write_text(dataset_fp + "\n")
        labeled.append(checkpoint)

    if not labeled:
        return ""

    latest = max(labeled, key=_checkpoint_step)
    pending["origin_step"] = origin_step
    pending["last_labeled_step"] = _checkpoint_step(latest)
    pending["last_checkpoint"] = latest.name
    pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2) + "\n")
    return str(latest)

def latest_checkpoint(exp_name):
    try:
        exp_name = validate_exp_name(exp_name)
    except ValueError:
        return ""
    exp = DDSP_ROOT / "exp" / exp_name
    pts = list(exp.glob("model_*.pt"))
    return str(max(pts, key=_checkpoint_step)) if pts else ""

def inspect_checkpoint(model_path):
    if not model_path:
        return "Чекпойнт не выбран."
    path = Path(model_path).expanduser()
    if not path.exists():
        return f"Файл не найден: {path}"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return f"Не удалось прочитать checkpoint: {exc}"
    if not isinstance(ckpt, dict):
        return f"Неожиданный формат checkpoint: {type(ckpt).__name__}"
    step = ckpt.get("global_step", "не указан")
    has_model = "model" in ckpt
    has_optimizer = ckpt.get("optimizer") is not None
    size_mb = path.stat().st_size / 2**20
    dataset_fp = checkpoint_dataset_fingerprint(path)
    return "\n".join([
        f"Файл: {path}",
        f"Шаг: {step}",
        f"Dataset ID: {dataset_fp[:12] if dataset_fp else 'неизвестен'}",
        f"Размер: {size_mb:.1f} МБ",
        f"Веса модели: {'есть' if has_model else 'нет'}",
        f"Optimizer: {'есть, полноценный resume' if has_optimizer else 'нет, продолжатся только веса и шаг'}",
    ])

def run_grader(exp_name):
    checks = []

    def add(status, name, detail):
        checks.append((status, name, detail))

    add("OK" if root_ok() else "ПРОБЛЕМА", "Исходники DDSP-SVC",
        str(DDSP_ROOT) if root_ok() else "train_reflow.py не найден")

    if torch.cuda.is_available():
        add("OK", "GPU/CUDA", f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    else:
        add("ПРОБЛЕМА", "GPU/CUDA", "CUDA недоступна. Для Colab выбери GPU runtime.")

    pretrained = [
        ("ContentVec", DDSP_ROOT / "pretrain/contentvec/pytorch_model.bin"),
        ("HiFiGAN model", DDSP_ROOT / "pretrain/nsf_hifigan/model"),
        ("HiFiGAN config", DDSP_ROOT / "pretrain/nsf_hifigan/config.json"),
        ("RMVPE", DDSP_ROOT / "pretrain/rmvpe/model.pt"),
    ]
    missing = [name for name, path in pretrained if not path.exists() or path.stat().st_size <= 0]
    add("OK" if not missing else "ПРОБЛЕМА", "Предобученные модели",
        "все найдены" if not missing else "нет или пустые: " + ", ".join(missing))

    train_wavs = len(list((DDSP_ROOT / "data/train/audio").glob("*.wav")))
    val_wavs = len(list((DDSP_ROOT / "data/val/audio").glob("*.wav")))
    dataset_ok = train_wavs > 0 and val_wavs > 0
    add("OK" if dataset_ok else "ПРОБЛЕМА", "Датасет",
        f"train {train_wavs}, val {val_wavs}")

    dataset_fp = current_dataset_fingerprint()
    add("OK" if dataset_fp else "ПРОБЛЕМА", "Dataset ID",
        dataset_fp[:12] if dataset_fp else "не найден")

    integrity = preprocessing_integrity()
    if integrity["ready"]:
        add("OK", "Целостность preprocessing",
            f"{integrity['complete']}/{integrity['total']} файлов готовы")
    else:
        detail = f"{integrity['complete']}/{integrity['total']} файлов готовы"
        if integrity["problems"]:
            detail += "; " + "; ".join(integrity["problems"][:2])
        add("ПРОБЛЕМА", "Целостность preprocessing", detail)

    try:
        source_cfg = yaml.safe_load((DDSP_ROOT / "configs/reflow.yaml").read_text())
        source_cfg["data"]["f0_extractor"] = "rmvpe" if torch.cuda.is_available() else "parselmouth"
        expected_manifest = preprocessing_manifest_for_config(source_cfg)
        manifest = read_preprocessing_manifest()
        manifest_ok = manifest == expected_manifest
        add("OK" if manifest_ok else "ПРОБЛЕМА", "Manifest preprocessing",
            "соответствует текущей среде" if manifest_ok else "отсутствует или параметры изменились")
    except Exception as exc:
        add("ПРОБЛЕМА", "Manifest preprocessing", f"не удалось проверить: {exc}")

    data_bytes = directory_size(DDSP_ROOT / "data")
    local_root = Path("/content") if Path("/content").exists() else DDSP_ROOT
    local_free = free_space(local_root)
    required_local = int(data_bytes * 1.2) + 1024 ** 3
    if local_free is None:
        add("ПРЕДУПРЕЖДЕНИЕ", "Локальное место", "не удалось определить")
    else:
        add("OK" if local_free >= required_local else "ПРОБЛЕМА", "Локальное место",
            f"свободно {human_bytes(local_free)}, для быстрого кэша желательно {human_bytes(required_local)}")

    try:
        exp_name = validate_exp_name(exp_name)
    except ValueError as exc:
        add("ПРОБЛЕМА", "Имя эксперимента", str(exc))
        exp_name = ""

    checkpoint = latest_checkpoint(exp_name) if exp_name else ""
    if not checkpoint:
        add("ПРЕДУПРЕЖДЕНИЕ", "Checkpoint", "ещё нет")
    else:
        path = Path(checkpoint)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict) or "model" not in ckpt or "global_step" not in ckpt:
                raise ValueError("нет обязательных ключей model/global_step")
            step = ckpt.get("global_step")
            has_optimizer = ckpt.get("optimizer") is not None
            add("OK", "Checkpoint",
                f"{path.name}, шаг {step}, {human_bytes(path.stat().st_size)}")
            add("OK" if has_optimizer else "ПРЕДУПРЕЖДЕНИЕ", "Optimizer checkpoint",
                "сохранён" if has_optimizer else "не сохранён, resume будет неполным")

            checkpoint_fp = checkpoint_dataset_fingerprint(path)
            if dataset_fp and checkpoint_fp == dataset_fp:
                add("OK", "Dataset ID checkpoint", checkpoint_fp[:12])
            elif not checkpoint_fp:
                add("ПРЕДУПРЕЖДЕНИЕ", "Dataset ID checkpoint",
                    "неизвестен; для legacy checkpoint используй явную привязку")
            else:
                add("ПРОБЛЕМА", "Dataset ID checkpoint",
                    f"checkpoint {checkpoint_fp[:12]}, текущий {dataset_fp[:12] if dataset_fp else 'нет'}")
        except Exception as exc:
            add("ПРОБЛЕМА", "Checkpoint", f"не удалось прочитать {path.name}: {exc}")

    pending = _pending_training_path(exp_name) if exp_name else None
    if pending and pending.exists() and not training_is_running():
        add("ПРЕДУПРЕЖДЕНИЕ", "Pending training",
            "найден незавершённый pending_dataset.json; он будет синхронизирован при следующем старте/статусе")
    else:
        add("OK", "Pending training", "активного stale pending нет")

    problems = sum(1 for status, _, _ in checks if status == "ПРОБЛЕМА")
    warnings = sum(1 for status, _, _ in checks if status == "ПРЕДУПРЕЖДЕНИЕ")
    if problems:
        verdict = f"Итог: есть проблемы ({problems}), предупреждений {warnings}."
    elif warnings:
        verdict = f"Итог: критических проблем нет, предупреждений {warnings}."
    else:
        verdict = "Итог: все проверки пройдены."

    lines = [verdict, ""]
    lines.extend(f"[{status}] {name}: {detail}" for status, name, detail in checks)
    return "\n".join(lines)

def bind_legacy_checkpoint_to_current_dataset(model_path):
    if training_is_running():
        return "Сначала останови обучение."
    if not model_path:
        return "Чекпойнт не выбран."

    path = Path(model_path).expanduser()
    if not path.exists():
        return f"Файл не найден: {path}"
    if path.suffix.lower() != ".pt":
        return "Нужен checkpoint model_*.pt."

    existing_fp = checkpoint_dataset_fingerprint(path)
    if existing_fp:
        return f"Checkpoint уже имеет Dataset ID {existing_fp[:12]}. Ничего не изменено."

    dataset_fp = current_dataset_fingerprint()
    if not dataset_fp:
        return "Текущий датасет не найден или не имеет Dataset ID."

    integrity = preprocessing_integrity()
    if not integrity["ready"]:
        return (
            "Текущий preprocessing не готов полностью. "
            f"Готово {integrity['complete']} из {integrity['total']} файлов. "
            "Сначала подготовь и проверь признаки."
        )

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return f"Checkpoint не удалось прочитать: {exc}"
    if not isinstance(ckpt, dict) or "model" not in ckpt or "global_step" not in ckpt:
        return "Файл не похож на checkpoint DDSP-SVC: нужны ключи model и global_step."

    sidecar = _checkpoint_dataset_sidecar(path)
    sidecar.write_text(dataset_fp + "\n")
    return (
        f"Legacy checkpoint привязан к текущему Dataset ID {dataset_fp[:12]}. "
        f"Файл: {path.name}, шаг: {ckpt.get('global_step')}."
    )

def do_inference(input_audio, model_path, key, formant, threshold, steps, method, t_start):
    if not input_audio:
        raise gr.Error("Нужен входной WAV.")
    model = Path(model_path).expanduser()
    if not model.exists():
        raise gr.Error("Файл модели не найден.")
    out_dir = DDSP_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    output = out_dir / f"result_{int(time.time())}.wav"
    cmd = [
        "python", "main_reflow.py",
        "-i", str(input_audio), "-m", str(model), "-o", str(output),
        "-k", str(key), "-f", str(formant), "-th", str(threshold),
        "-step", str(int(steps)), "-method", str(method), "-ts", str(t_start),
        "-pe", "rmvpe" if torch.cuda.is_available() else "parselmouth",
    ]
    p = subprocess.run(cmd, cwd=DDSP_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise gr.Error(p.stdout[-4000:])
    return str(output), p.stdout[-6000:]

class BrowserSvc:
    def __init__(self):
        self.model = None
        self.vocoder = None
        self.args = None
        self.units_encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, model_path):
        from reflow.vocoder import load_model_vocoder
        from ddsp.vocoder import Units_Encoder
        self.model, self.vocoder, self.args = load_model_vocoder(model_path, device=self.device)
        gate = self.args.data.cnhubertsoft_gate if self.args.data.encoder == "cnhubertsoftfish" else 10
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=gate,
            device=self.device,
        )

    def infer(self, audio, sr, pitch, formant, threshold, steps, method, t_start):
        from ddsp.vocoder import F0_Extractor, Volume_Extractor
        from ddsp.core import upsample
        hop = self.args.data.block_size * sr / self.args.data.sampling_rate
        win = self.args.data.volume_smooth_size * sr / self.args.data.sampling_rate
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        pitch_extractor = "rmvpe" if torch.cuda.is_available() else "parselmouth"
        f0_ext = F0_Extractor(pitch_extractor, sr, hop, 50.0, 1100.0)
        f0 = f0_ext.extract(audio, uv_interp=True, device=self.device)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch) / 12)
        formant_t = torch.tensor([[float(formant)]], device=self.device)
        volume_ext = Volume_Extractor(hop, win)
        volume_np = volume_ext.extract(audio)
        mask = (volume_np > 10 ** (float(threshold) / 20)).astype("float32")
        mask = torch.from_numpy(mask).to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume_np).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        units = self.units_encoder.encode(audio_t, sr, hop)
        spk_id = torch.LongTensor(np.array([[1]])).to(self.device)
        with torch.no_grad():
            out = self.model(
                units, f0, volume, spk_id=spk_id, spk_mix_dict=None,
                aug_shift=formant_t, vocoder=self.vocoder, infer=True,
                return_wav=True, infer_step=int(steps), method=str(method),
                t_start=float(t_start), silence_front=0, use_tqdm=False,
            )
            out *= mask[:, -out.shape[-1]:]
        return out.squeeze().detach().cpu().numpy().astype(np.float32), self.args.data.sampling_rate

def realtime_convert(chunk, model_path, pitch, formant, threshold, steps, method, t_start, context_seconds):
    global _RT_MODEL, _RT_MODEL_PATH, _RT_BUFFER
    if chunk is None:
        return None
    sr, audio = chunk
    if audio is None or len(audio) == 0:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.5:
        audio /= 32768.0
    if int(sr) != 44100:
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=44100)
        sr = 44100
    model_path = str(Path(model_path).expanduser())
    if _RT_MODEL is None or _RT_MODEL_PATH != model_path:
        _RT_MODEL = BrowserSvc()
        _RT_MODEL.load(model_path)
        _RT_MODEL_PATH = model_path
        _RT_BUFFER = np.zeros(0, dtype=np.float32)
    max_len = int(float(context_seconds) * sr)
    _RT_BUFFER = np.concatenate([_RT_BUFFER, audio])[-max_len:]
    out, out_sr = _RT_MODEL.infer(_RT_BUFFER, sr, pitch, formant, threshold, steps, method, t_start)
    take = max(1, int(len(audio) * out_sr / sr))
    return out_sr, out[-take:]

def reset_realtime():
    global _RT_MODEL, _RT_MODEL_PATH, _RT_BUFFER
    _RT_MODEL = None
    _RT_MODEL_PATH = None
    _RT_BUFFER = np.zeros(0, dtype=np.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Realtime состояние сброшено."

with gr.Blocks(title="DDSP-SVC ReFlow для Google Colab") as demo:
    gr.Markdown("# DDSP-SVC ReFlow\nРусский интерфейс для подготовки датасета, обучения, обычной генерации и экспериментального realtime из браузера.")
    with gr.Tab("Проверка"):
        status = gr.Textbox(label="Состояние системы", lines=8, value=system_status())
        gr.Button("Обновить состояние").click(system_status, outputs=status)
    with gr.Tab("Датасет"):
        dataset_zip = gr.File(label="ZIP с WAV файлами", file_types=[".zip"], type="filepath")
        val_count = gr.Slider(1, 30, value=10, step=1, label="Файлов для проверки")
        dataset_status = gr.Textbox(label="Результат")
        gr.Button("Подготовить датасет").click(prepare_dataset, [dataset_zip, val_count], dataset_status)
    with gr.Tab("Обучение"):
        batch = gr.Slider(1, 48, value=32, step=1, label="Batch size. Для T4 начни с 32")
        cache = gr.Checkbox(value=False, label="Держать весь датасет в памяти")
        exp_name = gr.Textbox(value="reflow-colab", label="Имя эксперимента")
        epochs = gr.Number(value=100000, precision=0, minimum=1, label="Количество эпох")
        interval_val = gr.Number(value=2000, precision=0, minimum=1, label="Проверка и новый чекпойнт каждые N шагов")
        interval_force_save = gr.Number(value=10000, precision=0, minimum=1, label="Оставлять постоянный чекпойнт каждые N шагов")
        fast_local_data = gr.Checkbox(
            value=True,
            label="Быстрый локальный кэш для обучения. Признаки останутся на Google Drive, но обучение будет читать их из /content",
        )
        save_optimizer = gr.Checkbox(
            value=True,
            label="Сохранять optimizer для полноценного продолжения обучения после новой сессии",
        )
        allow_dataset_change = gr.Checkbox(
            value=False,
            label="Разрешить fine-tune существующего checkpoint на другом датасете",
        )
        train_log = gr.Textbox(label="Журнал", lines=18)
        preprocess_inputs = [batch, cache, exp_name, epochs, interval_val, interval_force_save]
        training_inputs = preprocess_inputs + [fast_local_data, save_optimizer, allow_dataset_change]
        gr.Button("Подготовить признаки").click(run_preprocess, preprocess_inputs, train_log)
        gr.Button("Запустить обучение").click(start_training, training_inputs, train_log)
        gr.Button("Показать статус").click(training_status, outputs=train_log)
        gr.Button("Остановить обучение").click(stop_training, outputs=train_log)
        last_ckpt = gr.Textbox(label="Последний чекпойнт")
        ckpt_info = gr.Textbox(label="Проверка checkpoint", lines=6)
        gr.Button("Найти последний чекпойнт").click(latest_checkpoint, exp_name, last_ckpt)
        gr.Button("Проверить checkpoint").click(inspect_checkpoint, last_ckpt, ckpt_info)
        gr.Button("Привязать legacy checkpoint к текущему Dataset ID").click(
            bind_legacy_checkpoint_to_current_dataset,
            last_ckpt,
            ckpt_info,
        )
    with gr.Tab("Грейдер"):
        gr.Markdown(
            "Проверяет готовность текущей среды, датасета, preprocessing и выбранного эксперимента. "
            "Числового рейтинга нет: каждая проблема показывается отдельно."
        )
        grader_exp_name = gr.Textbox(value="reflow-colab", label="Имя эксперимента для проверки")
        grader_output = gr.Textbox(label="Результат грейдера", lines=22)
        gr.Button("Запустить грейдер").click(run_grader, grader_exp_name, grader_output)
    with gr.Tab("Генерация"):
        input_audio = gr.Audio(label="Исходный голос", type="filepath")
        model_path = gr.Textbox(label="Путь к model_*.pt", placeholder="/content/drive/MyDrive/...")
        key = gr.Slider(-24, 24, value=0, step=1, label="Высота, полутонов")
        formant = gr.Slider(-2, 2, value=0, step=0.05, label="Сдвиг формант")
        threshold = gr.Slider(-60, 0, value=-45, step=1, label="Порог тишины, дБ")
        steps = gr.Slider(1, 50, value=20, step=1, label="Шагов ReFlow")
        method = gr.Radio(["euler", "rk4"], value="euler", label="Метод")
        t_start = gr.Slider(0, 0.95, value=0.7, step=0.05, label="t_start")
        output_audio = gr.Audio(label="Результат", type="filepath")
        infer_log = gr.Textbox(label="Журнал генерации", lines=10)
        gr.Button("Преобразовать").click(
            do_inference,
            [input_audio, model_path, key, formant, threshold, steps, method, t_start],
            [output_audio, infer_log],
        )
    with gr.Tab("Realtime"):
        gr.Markdown("Экспериментальный режим для микрофона браузера. Чем больше контекст, тем стабильнее голос, но выше задержка.")
        rt_model = gr.Textbox(label="Путь к model_*.pt")
        mic = gr.Audio(label="Микрофон", sources=["microphone"], streaming=True)
        rt_pitch = gr.Slider(-24, 24, value=0, step=1, label="Высота, полутонов")
        rt_formant = gr.Slider(-2, 2, value=0, step=0.05, label="Сдвиг формант")
        rt_threshold = gr.Slider(-60, 0, value=-45, step=1, label="Порог тишины, дБ")
        rt_steps = gr.Slider(1, 30, value=8, step=1, label="Шагов ReFlow")
        rt_method = gr.Radio(["euler", "rk4"], value="euler", label="Метод")
        rt_t_start = gr.Slider(0, 0.95, value=0.7, step=0.05, label="t_start")
        rt_context = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Контекст, секунд")
        rt_out = gr.Audio(label="Поток после преобразования", streaming=True, autoplay=True)
        rt_state = gr.Textbox(label="Состояние realtime")
        mic.stream(
            realtime_convert,
            [mic, rt_model, rt_pitch, rt_formant, rt_threshold, rt_steps, rt_method, rt_t_start, rt_context],
            rt_out,
            stream_every=0.5,
            time_limit=3600,
        )
        gr.Button("Сбросить realtime").click(reset_realtime, outputs=rt_state)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
