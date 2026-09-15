from pathlib import Path
import numpy as np
import soundfile as sf

root = Path("/workspace/DDSP-SVC")
sr = 44100
for split, count in [("train", 2), ("val", 1)]:
    out = root / "data" / split / "audio"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        seconds = 2.2
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        f = 120.0 + i * 25.0
        x = 0.10 * np.sin(2 * np.pi * f * t)
        x += 0.02 * np.sin(2 * np.pi * 2 * f * t)
        x += 0.002 * np.random.default_rng(i).standard_normal(t.shape).astype(np.float32)
        sf.write(out / f"smoke_{i}.wav", x, sr)
print("Smoke dataset ready")
