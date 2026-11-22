"""Generate synthetic inputs without reading or sampling clinical records."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MUSCLES = ["Biceps brachii", "Extensor carpi radialis", "Triceps brachii",
           "Flexor digitorum superficialis", "Dorsal interosseous", "Iliopsoas",
           "Quadriceps", "Tibialis anterior", "Extensor hallucis longus", "Gastrocnemius"]
SEGMENTS = ([f"C{i}" for i in range(2, 9)] + [f"T{i}" for i in range(1, 13)]
            + [f"L{i}" for i in range(1, 6)] + ["S1", "S2", "S3", "S4_5"])

def generate(n=120, seed=123):
    rng = np.random.default_rng(seed)
    data = {"Patient ID": [f"SYN{i:04d}" for i in range(n)]}
    for muscle in MUSCLES:
        for side in ("R", "L"):
            data[f"2 weeks::{muscle}_{side}"] = rng.integers(0, 6, n)
    for modality in ("T", "A"):
        for segment in SEGMENTS:
            for side in ("R", "L"):
                data[f"2 weeks::{modality}{side}_{segment}"] = rng.integers(0, 3, n)
    data["6 months later::ASIAImpairmentScale"] = rng.permutation(np.resize([1, 2, 3, 4], n))
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "example.xlsx")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.n < 24:
        parser.error("--n must be at least 24")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.n, args.seed).to_excel(args.output, index=False)
    print(f"Synthetic dataset saved to {args.output}")
