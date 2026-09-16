import os
import sys
import hashlib
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
_RT_MODEL = None
_RT_MODEL_PATH = None
_RT_BUFFER = np.zeros(0, dtype=np.float32)

def root_ok():
    return (DDSP_ROOT / "train_reflow.py").exists()

def tail_text(path, lines=80):
    p = Path(path)
    if not p.exists():
        return "Журнал пока пуст."
    data = p.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])

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

def make_config(batch_size=32, cache_all=False, exp_name="reflow-colab", epochs=100000, interval_val=2000, interval_force_save=10000, fast_local_data=False, save_optimizer=True):
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
    preprocess_ready = (
        (DDSP_ROOT / "data/train/pitch_aug_dict.npy").exists()
        and (DDSP_ROOT / "data/val/pitch_aug_dict.npy").exists()
    )
    exp_root = DDSP_ROOT / "exp"
    checkpoints = list(exp_root.glob("*/model_*.pt")) if exp_root.exists() else []
    last_ckpt = max(checkpoints, key=_checkpoint_step) if checkpoints else None

    parts = [
        f"Папка DDSP-SVC: {DDSP_ROOT}",
        f"Исходники: {'найдены' if root_ok() else 'не найдены'}",
        f"PyTorch: {torch.__version__}",
        f"GPU: {gpu}",
        f"Датасет: train {train_wavs} WAV, val {val_wavs} WAV",
        f"Preprocessing: {'готов' if preprocess_ready else 'не готов'}",
        f"Сегмент обучения: {training_segment_duration():.3f} с",
        f"Последний checkpoint: {last_ckpt.name if last_ckpt else 'нет'}",
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
    cfg = make_config(batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save)
    cmd = ["python", "preprocess.py", "-c", str(cfg), "-j", "2"]
    p = subprocess.run(cmd, cwd=DDSP_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout[-12000:]
    if p.returncode != 0:
        raise gr.Error("Preprocessing завершился с ошибкой.\n" + output[-5000:])
    return output

def start_training(batch_size, cache_all, exp_name, epochs, interval_val, interval_force_save, fast_local_data, save_optimizer):
    global TRAIN_PROCESS, TRAIN_LOG
    if TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None:
        return "Обучение уже идёт."

    source = DDSP_ROOT / "data"
    preprocess_ready = (
        (source / "train/pitch_aug_dict.npy").exists()
        and (source / "val/pitch_aug_dict.npy").exists()
    )
    if not preprocess_ready:
        return "Сначала нажми «Подготовить признаки». Готовый preprocessing не найден."

    resume_from = latest_checkpoint(exp_name)

    if fast_local_data:
        local_data = Path(os.environ.get("DDSP_LOCAL_DATA", "/content/ddsp-local-data")).resolve()
        shutil.rmtree(local_data, ignore_errors=True)
        local_data.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "train", local_data / "train")
        shutil.copytree(source / "val", local_data / "val")

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
    if resume_from:
        resume_note = f"Продолжаем с {Path(resume_from).name}."
    else:
        resume_note = "Готового чекпойнта нет, начинаем с нуля."
    optimizer_note = "Состояние optimizer сохраняется." if save_optimizer else "Optimizer не сохраняется."
    return f"Обучение запущено. PID {TRAIN_PROCESS.pid}. {resume_note} {optimizer_note} Кнопка Статус покажет последние строки."

def training_status():
    global TRAIN_PROCESS
    state = "не запускалось"
    if TRAIN_PROCESS is not None:
        code = TRAIN_PROCESS.poll()
        state = "идёт" if code is None else f"завершено, код {code}"
    log = tail_text(TRAIN_LOG) if TRAIN_LOG else "Журнал пока пуст."
    return f"Состояние: {state}\n\n{log}"

def stop_training():
    global TRAIN_PROCESS
    if TRAIN_PROCESS is None or TRAIN_PROCESS.poll() is not None:
        return "Активного обучения нет."
    TRAIN_PROCESS.terminate()
    try:
        TRAIN_PROCESS.wait(timeout=10)
    except subprocess.TimeoutExpired:
        TRAIN_PROCESS.kill()
    return "Обучение остановлено. Чекпойнты, которые успели сохраниться, не удалены."

def _checkpoint_step(path):
    try:
        return int(Path(path).stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1

def latest_checkpoint(exp_name):
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
    return "\n".join([
        f"Файл: {path}",
        f"Шаг: {step}",
        f"Размер: {size_mb:.1f} МБ",
        f"Веса модели: {'есть' if has_model else 'нет'}",
        f"Optimizer: {'есть, полноценный resume' if has_optimizer else 'нет, продолжатся только веса и шаг'}",
    ])

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
        train_log = gr.Textbox(label="Журнал", lines=18)
        preprocess_inputs = [batch, cache, exp_name, epochs, interval_val, interval_force_save]
        training_inputs = preprocess_inputs + [fast_local_data, save_optimizer]
        gr.Button("Подготовить признаки").click(run_preprocess, preprocess_inputs, train_log)
        gr.Button("Запустить обучение").click(start_training, training_inputs, train_log)
        gr.Button("Показать статус").click(training_status, outputs=train_log)
        gr.Button("Остановить обучение").click(stop_training, outputs=train_log)
        last_ckpt = gr.Textbox(label="Последний чекпойнт")
        ckpt_info = gr.Textbox(label="Проверка checkpoint", lines=6)
        gr.Button("Найти последний чекпойнт").click(latest_checkpoint, exp_name, last_ckpt)
        gr.Button("Проверить checkpoint").click(inspect_checkpoint, last_ckpt, ckpt_info)
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
