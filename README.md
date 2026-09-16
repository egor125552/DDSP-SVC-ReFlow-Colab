# DDSP-SVC ReFlow Colab

Готовый Google Colab набор для актуального DDSP-SVC ReFlow.

Что внутри:

- русский Gradio интерфейс;
- подготовка ZIP датасета и приведение WAV к 44,1 кГц mono;
- preprocessing;
- запуск и остановка обучения;
- настройка batch size, количества эпох и интервалов чекпойнтов;
- просмотр журнала обучения;
- поиск последнего чекпойнта;
- обычное преобразование WAV;
- экспериментальный realtime из микрофона браузера;
- сохранение папки exp в Google Drive;
- CPU Docker smoke test для проверки зависимостей без видеокарты.

## Google Colab

Открой DDSP_SVC_ReFlow_Colab.ipynb в Google Colab и выбери GPU runtime.

Прямая ссылка: https://colab.research.google.com/github/egor125552/DDSP-SVC-ReFlow-Colab/blob/main/DDSP_SVC_ReFlow_Colab.ipynb

Для Tesla T4 интерфейс начинает с batch size 32. Если памяти хватает, можно поднять. Если появляется CUDA out of memory, уменьши batch size.

Перед запуском Gradio блокнот выполняет GPU smoke test: проверяет CUDA, показывает модель GPU и VRAM, делает короткое FP16 вычисление и отдельно загружает RMVPE на GPU. Если Colab выдал CPU, блокнот остановится с понятным сообщением вместо того, чтобы начать мучительно медленное обучение.

## Realtime

Оригинальный DDSP-SVC realtime использует локальные аудиоустройства через sounddevice. Это не работает напрямую в Colab, потому что микрофон находится в браузере пользователя, а Python работает на удалённой машине.

Поэтому здесь есть отдельный экспериментальный браузерный realtime через Gradio. Он хранит контекст предыдущего звука и отправляет поток на модель. Это не равно локальному gui_reflow.py по минимальной задержке.

Сам realtime обработчик проверен на Linux и выдаёт корректный поток 44,1 кГц. Настоящий микрофон браузера и реальная задержка на Tesla T4 требуют проверки уже в запущенном Colab. Python gradio_client 2.7 имеет отдельную проблему со streaming endpoint, поэтому для realtime ориентируйся на браузерный интерфейс, а не на gradio_client.

## Проверено на Linux

В CPU Docker smoke test реально прошли установка зависимостей, подготовка датасета, preprocessing, два шага обучения, сохранение чекпойнта, повторная загрузка модели, обычный inference до WAV и прямой вызов realtime обработчика. Для CPU автоматически используется parselmouth, а на CUDA остаётся RMVPE.

## Docker smoke test

Dockerfile.cpu-smoke специально использует CPU PyTorch. Он нужен не для качественного обучения, а чтобы на Linux без GPU проверить установку зависимостей, preprocessing и короткий запуск обучения.

Сборка:

docker build -f Dockerfile.cpu-smoke -t ddsp-svc-reflow-smoke .

После сборки сначала скачай предобученные файлы:

docker run --rm -v ddsp-pretrain:/workspace/DDSP-SVC/pretrain ddsp-svc-reflow-smoke bash /workspace/tools/download_pretrained.sh

Затем можно запускать smoke test с тем же volume.
