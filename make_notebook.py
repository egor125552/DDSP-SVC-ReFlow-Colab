import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }

cells = [
    md("""# DDSP-SVC ReFlow для Google Colab

Этот блокнот ставит актуальный DDSP-SVC, загружает обязательные предобученные модели и запускает русский Gradio интерфейс.

Перед запуском выбери GPU runtime. Для Tesla T4 стартовый batch size в интерфейсе установлен 32."""),
    code("""from google.colab import drive
drive.mount('/content/drive')
print('Google Drive подключён')"""),
    code("""!apt-get -qq update
!apt-get -qq install -y ffmpeg libsndfile1 unzip
!rm -rf /content/DDSP-SVC /content/DDSP-SVC-ReFlow-Colab
!git clone --depth 1 https://github.com/yxlllc/DDSP-SVC.git /content/DDSP-SVC
!git clone --depth 1 https://github.com/egor125552/DDSP-SVC-ReFlow-Colab.git /content/DDSP-SVC-ReFlow-Colab
%cd /content/DDSP-SVC
!python -m pip install -q --upgrade pip
!python -m pip install -q -r requirements.txt
!python -m pip install -q -r /content/DDSP-SVC-ReFlow-Colab/requirements-extra.txt
print('Зависимости установлены')"""),
    code("""import os, shutil
from pathlib import Path
import torch

print('PyTorch:', torch.__version__)
print('CUDA доступна:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM, ГБ:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))

drive_root = Path('/content/drive/MyDrive/DDSP-SVC-ReFlow')
drive_root.mkdir(parents=True, exist_ok=True)
exp_link = Path('/content/DDSP-SVC/exp')
if exp_link.exists() or exp_link.is_symlink():
    if exp_link.is_symlink() or exp_link.is_file():
        exp_link.unlink()
    else:
        shutil.rmtree(exp_link)
os.symlink(drive_root / 'exp', exp_link, target_is_directory=True)
(drive_root / 'exp').mkdir(exist_ok=True)
print('Чекпойнты будут сохраняться в', drive_root / 'exp')"""),
    code("""!bash /content/DDSP-SVC-ReFlow-Colab/scripts/download_pretrained.sh /content/DDSP-SVC
print('Предобученные модели готовы')"""),
    code("""import os
os.environ['DDSP_ROOT'] = '/content/DDSP-SVC'
%cd /content/DDSP-SVC
!python /content/DDSP-SVC-ReFlow-Colab/colab_app.py"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("DDSP_SVC_ReFlow_Colab.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2))
print(out)
