"""Task 4: compare raw MC, calibrated MC and TB validation efficiencies."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformer"))

from predict import load_model
from ot_model import NeuralOT


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ckpt_path = (
        HERE.parent / "transformer/checkpoints/pid_transformer/best.pt"
    )
    mask_path = HERE / "derived/tb_is_derivation.npy"
    out = HERE / "results/task4_efficiency"

    if out.exists():
        raise FileExistsError(
            f"{out} esiste già. Conserva i risultati prima di rieseguire."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, _, _, _ = load_model(ckpt_path, device)
    model.eval()
    model.requires_grad_(False)

    checkpoint_hash = sha256_file(ckpt_path)
    mask_hash = sha256_file(mask_path)

    with np.load(HERE / "derived/latents_MC.npz") as data:
        if data["classes"].tolist() != classes:
            raise ValueError("Ordine delle classi MC diverso dal checkpoint.")
        if str(data["checkpoint_sha256"].item()) != checkpoint_hash:
            raise ValueError("Checkpoint MC diverso da quello caricato.")

        z_mc = data["z_enc"]
        logits_mc = data["logits"]
        labels_mc = data["label"]
        energy_mc = data["energy_mev"]
        rows_mc = data["parquet_row"]

    with np.load(HERE / "derived/latents_TB.npz") as data:
        if data["classes"].tolist() != classes:
            raise ValueError("Ordine delle classi TB diverso dal checkpoint.")
        if str(data["checkpoint_sha256"].item()) != checkpoint_hash:
            raise ValueError("Checkpoint TB diverso da quello caricato.")

        logits_tb = data["logits"]
        labels_tb = data["label"]
        energy_tb = data["energy_mev"]
        split = data["is_derivation"]
        rows_tb = data["parquet_row"]

    if not np.array_equal(rows_mc, np.arange(len(labels_mc))):
        raise ValueError("Ordine delle righe MC inatteso.")
    if not np.array_equal(rows_tb, np.arange(len(labels_tb))):
        raise ValueError("Ordine delle righe TB inatteso.")
    if split.dtype != np.bool_:
        raise ValueError("La maschera TB deve essere booleana.")
    if not np.array_equal(split, np.load(mask_path)):
        raise ValueError("La maschera TB non coincide con quella originale.")

    # Verify that the loaded frozen head reproduces the stored raw logits.
    check_rows = np.linspace(
        0, len(z_mc) - 1, num=min(512, len(z_mc)), dtype=int
    )
    with torch.no_grad():
        check_logits = model.head_from_latent(
            torch.from_numpy(z_mc[check_rows]).to(device)
        ).cpu().numpy()

    np.testing.assert_allclose(
        check_logits, logits_mc[check_rows], rtol=1e-4, atol=1e-5
    )
    print("PASS: frozen head matches stored raw logits.", flush=True)

    # Preflight checks for every map before processing events.
    configurations = {}
    for cls in ["e", "p", "C"]:
        folder = HERE / f"results/task3_{cls}_h128_1000"
        config = json.loads((folder / "config.json").read_text())
        if config["class_name"] != cls:
            raise ValueError("Classe della mappa non corretta.")
        if config["classes"] != classes:
            raise ValueError("Ordine delle classi della mappa non corretto.")
        if config["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("La mappa usa un checkpoint PID diverso.")
        if config["tb_split_sha256"] != mask_hash:
            raise ValueError("La mappa usa uno split TB diverso.")
        if not (folder / "map.pt").is_file():
            raise FileNotFoundError(folder / "map.pt")
        configurations[cls] = config

    out.mkdir(parents=True)
    summary = []

    for cls in ["e", "p", "C"]:
        class_index = classes.index(cls)
        config = configurations[cls]
        common = np.asarray(config["common_energies"])
        folder = HERE / f"results/task3_{cls}_h128_1000"

        mc_rows = np.flatnonzero(
            (labels_mc == class_index) & np.isin(energy_mc, common)
        )
        tb_rows = np.flatnonzero(
            (labels_tb == class_index)
            & ~split
            & np.isin(energy_tb, common)
        )

        if len(mc_rows) == 0 or len(tb_rows) == 0:
            raise ValueError(f"Campione vuoto per {cls}.")

        transport = NeuralOT.load(folder / "map.pt", device=device)
        calibrated_z = np.empty((len(mc_rows), 64), dtype=np.float32)
        calibrated_logits = np.empty(
            (len(mc_rows), len(classes)), dtype=np.float32
        )

        print(
            f"\n{cls}: MC={len(mc_rows):,}; "
            f"TB validation={len(tb_rows):,}",
            flush=True,
        )

        for start in range(0, len(mc_rows), 4096):
            stop = min(start + 4096, len(mc_rows))
            selected = mc_rows[start:stop]

            # OT needs input gradients. Do not wrap this in no_grad.
            mapped = transport.transform(z_mc[selected], batch_size=256)

            # transform already restores target-domain latent coordinates.
            # No further normalisation is needed before the frozen head.
            with torch.no_grad():
                logits = model.head_from_latent(
                    torch.from_numpy(mapped).to(device)
                ).cpu().numpy()

            if not np.isfinite(logits).all():
                raise ValueError("Logits calibrati non finiti.")

            calibrated_z[start:stop] = mapped
            calibrated_logits[start:stop] = logits

            if start == 0 or stop // 4096 % 25 == 0 or stop == len(mc_rows):
                print(f"  Processed {stop:,}/{len(mc_rows):,}", flush=True)

        raw_correct = int(
            (logits_mc[mc_rows].argmax(1) == class_index).sum()
        )
        cal_correct = int(
            (calibrated_logits.argmax(1) == class_index).sum()
        )
        tb_correct = int(
            (logits_tb[tb_rows].argmax(1) == class_index).sum()
        )

        raw = raw_correct / len(mc_rows)
        calibrated = cal_correct / len(mc_rows)
        tb = tb_correct / len(tb_rows)

        original_gap = abs(raw - tb)
        residual = abs(calibrated - tb)

        summary.append({
            "cls": cls,
            "n_MC": len(mc_rows),
            "n_TB_validation": len(tb_rows),
            "correct_raw_MC": raw_correct,
            "correct_calibrated_MC": cal_correct,
            "correct_TB": tb_correct,
            "raw_MC": raw,
            "calibrated_MC": calibrated,
            "TB_validation": tb,
            "original_gap_pp": 100 * original_gap,
            "residual_pp": 100 * residual,
            "residual_over_original_gap": (
                residual / original_gap if original_gap > 0 else np.nan
            ),
        })

        np.savez(
            out / f"calibrated_MC_{cls}.npz",
            z_enc=calibrated_z,
            logits=calibrated_logits,
            label=labels_mc[mc_rows],
            energy_mev=energy_mc[mc_rows],
            parquet_row=rows_mc[mc_rows],
            classes=np.asarray(classes),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            map_sha256=np.asarray(sha256_file(folder / "map.pt")),
        )

        pd.DataFrame(summary).to_csv(
            out / "efficiency_comparison.csv", index=False
        )

        print(
            f"  raw MC={raw:.6f}; calibrated MC={calibrated:.6f}; "
            f"TB validation={tb:.6f}",
            flush=True,
        )

    print("\nEfficiency comparison:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()