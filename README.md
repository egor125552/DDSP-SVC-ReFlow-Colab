# DDSP-SVC ReFlow Colab

Готовый Google Colab набор для актуального DDSP-SVC ReFlow.

Что внутри:

- русский Gradio интерфейс;
- подготовка ZIP датасета и приведение WAV к 44,1 кГц mono;
- preprocessing;
- запуск и остановка обучения;
- просмотр журнала обучения;
- поиск последнего чекпойнта;
- обычное преобразование WAV;
- экспериментальный realtime из микрофона браузера;
- сохранение папки exp в Google Drive;
- CPU Docker smoke test для проверки зависимостей без видеокарты.

## Google Colab

Открой DDSP_SVC_ReFlow_Colab.ipynb в Google Colab и выбери GPU runtime.

Для Tesla T4 интерфейс начинает с batch size 32. Если памяти хватает, можно поднять. Если появляется CUDA out of memory, уменьши batch size.

## Realtime

Оригинальный DDSP-SVC realtime использует локальные аудиоустройства через sounddevice. Это не работает напрямую в Colab, потому что микрофон находится в браузере пользователя, а Python работает на удалённой машине.

Поэтому здесь есть отдельный экспериментальный браузерный realtime через Gradio. Он хранит контекст предыдущего звука и отправляет поток на модель. Это не равно локальному gui_reflow.py по минимальной задержке.

## Docker smoke test

Dockerfile.cpu-smoke специально использует CPU PyTorch. Он нужен не для качественного обучения, а чтобы на Linux без GPU проверить установку зависимостей, preprocessing и короткий запуск обучения.

Сборка:

docker build -f Dockerfile.cpu-smoke -t ddsp-svc-reflow-smoke .

После сборки сначала скачай предобученные файлы:

docker run --rm -v ddsp-pretrain:/workspace/DDSP-SVC/pretrain ddsp-svc-reflow-smoke bash /workspace/tools/download_pretrained.sh

Затем можно запускать smoke test с тем же volume.
