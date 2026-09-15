"""Task 6: evaluate energy-conditioned OT with the frozen PID head."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformer"))

from predict import load_model
from train_conditional_ot import ConditionalTransport
from train_ot import sha256_file


OUT = HERE / "results/task6_efficiency"


def main():
    if OUT.exists():
        raise FileExistsError(
            f"{OUT} esiste già. Non sovrascrivo i risultati."
        )

    ckpt_path = (
        HERE.parent / "transformer/checkpoints/pid_transformer/best.pt"
    )
    mask_path = HERE / "derived/tb_is_derivation.npy"
    baseline_path = (
        HERE / "results/task4_efficiency/efficiency_comparison.csv"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, _, _, _ = load_model(ckpt_path, device)
    model.eval()
    model.requires_grad_(False)

    checkpoint_hash = sha256_file(ckpt_path)
    mask_hash = sha256_file(mask_path)
    baseline = pd.read_csv(baseline_path).set_index("cls")

    with np.load(HERE / "derived/latents_MC.npz") as data:
        if data["classes"].tolist() != classes:
            raise ValueError("Classi MC diverse dal checkpoint.")
        if str(data["checkpoint_sha256"].item()) != checkpoint_hash:
            raise ValueError("Checkpoint MC diverso.")

        z_mc = data["z_enc"]
        logits_mc = data["logits"]
        labels_mc = data["label"]
        energy_mc = data["energy_mev"]
        rows_mc = data["parquet_row"]

    with np.load(HERE / "derived/latents_TB.npz") as data:
        if data["classes"].tolist() != classes:
            raise ValueError("Classi TB diverse dal checkpoint.")
        if str(data["checkpoint_sha256"].item()) != checkpoint_hash:
            raise ValueError("Checkpoint TB diverso.")

        logits_tb = data["logits"]
        labels_tb = data["label"]
        energy_tb = data["energy_mev"]
        split = data["is_derivation"]
        rows_tb = data["parquet_row"]

    np.testing.assert_array_equal(
        rows_mc, np.arange(len(labels_mc))
    )
    np.testing.assert_array_equal(
        rows_tb, np.arange(len(labels_tb))
    )
    if split.dtype != np.bool_ or split.shape != labels_tb.shape:
        raise ValueError("Maschera TB non valida.")
    np.testing.assert_array_equal(split, np.load(mask_path))

    # Check the unchanged PID head on events spanning the MC file.
    check_rows = np.linspace(
        0, len(z_mc) - 1, min(512, len(z_mc)), dtype=int
    )
    with torch.no_grad():
        check_logits = model.head_from_latent(
            torch.from_numpy(z_mc[check_rows]).to(device)
        ).cpu().numpy()

    np.testing.assert_allclose(
        check_logits, logits_mc[check_rows],
        rtol=1e-4, atol=1e-5,
    )
    print("PASS: frozen head matches stored raw logits.", flush=True)

    # Check all maps and event selections before writing results.
    selections = {}

    for cls in ["e", "p", "C"]:
        folder = HERE / f"results/task6_{cls}_h128_1000"
        map_path = folder / "map.pt"
        config = json.loads(
            (folder / "config.json").read_text(encoding="utf-8")
        )

        if not map_path.is_file():
            raise FileNotFoundError(map_path)
        if config["class_name"] != cls or config["classes"] != classes:
            raise ValueError("Classe della mappa non corretta.")
        if config["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint PID della mappa diverso.")
        if config["tb_split_sha256"] != mask_hash:
            raise ValueError("Split TB della mappa diverso.")

        idx = classes.index(cls)
        common = np.asarray(config["common_energies"])

        mc_rows = np.flatnonzero(
            (labels_mc == idx) & np.isin(energy_mc, common)
        )
        tb_rows = np.flatnonzero(
            (labels_tb == idx)
            & ~split
            & np.isin(energy_tb, common)
        )

        # Compare the actual MC rows with the previous calibration.
        previous_path = (
            HERE / "results/task4_efficiency"
            / f"calibrated_MC_{cls}.npz"
        )
        with np.load(previous_path) as previous:
            np.testing.assert_array_equal(
                previous["parquet_row"], mc_rows
            )

        if len(mc_rows) == 0 or len(tb_rows) == 0:
            raise ValueError(f"Campione vuoto per {cls}.")

        raw_correct = int(
            (logits_mc[mc_rows].argmax(1) == idx).sum()
        )
        tb_correct = int(
            (logits_tb[tb_rows].argmax(1) == idx).sum()
        )

        for key, value in {
            "n_MC": len(mc_rows),
            "n_TB_validation": len(tb_rows),
            "correct_raw_MC": raw_correct,
            "correct_TB": tb_correct,
        }.items():
            if value != int(baseline.loc[cls, key]):
                raise ValueError(f"{cls}: {key} diverso dal Task 4.")

        selections[cls] = (
            map_path, mc_rows, tb_rows, raw_correct, tb_correct
        )

    OUT.mkdir(parents=True)
    summary = []

    for cls in ["e", "p", "C"]:
        idx = classes.index(cls)
        map_path, mc_rows, tb_rows, raw_correct, tb_correct = selections[cls]

        transport = ConditionalTransport.load(map_path, device=device)

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

            # Pass the physical beam energy. The map normalises it.
            # This operation needs gradients with respect to the latents.
            mapped = transport.transform(
                z_mc[selected],
                energy_mc[selected],
                batch_size=256,
            )

            # transform returns original target latent coordinates.
            with torch.no_grad():
                logits = model.head_from_latent(
                    torch.from_numpy(mapped).to(device)
                ).cpu().numpy()

            if not np.isfinite(logits).all():
                raise ValueError("Logits calibrati non finiti.")

            calibrated_z[start:stop] = mapped
            calibrated_logits[start:stop] = logits

            if start == 0 or stop % 102400 == 0 or stop == len(mc_rows):
                print(
                    f"  Processed {stop:,}/{len(mc_rows):,}",
                    flush=True,
                )

        conditional_correct = int(
            (calibrated_logits.argmax(1) == idx).sum()
        )

        raw = raw_correct / len(mc_rows)
        conditional = conditional_correct / len(mc_rows)
        tb = tb_correct / len(tb_rows)
        unconditioned = float(baseline.loc[cls, "calibrated_MC"])

        original_gap = abs(raw - tb)
        old_residual = abs(unconditioned - tb)
        new_residual = abs(conditional - tb)

        summary.append({
            "cls": cls,
            "n_MC": len(mc_rows),
            "n_TB_validation": len(tb_rows),
            "correct_raw_MC": raw_correct,
            "correct_conditional_MC": conditional_correct,
            "correct_TB": tb_correct,
            "raw_MC": raw,
            "unconditional_MC": unconditioned,
            "conditional_MC": conditional,
            "TB_validation": tb,
            "original_gap_pp": 100 * original_gap,
            "unconditional_residual_pp": 100 * old_residual,
            "conditional_residual_pp": 100 * new_residual,
            "improvement_vs_unconditional_pp": (
                100 * (old_residual - new_residual)
            ),
            "conditional_residual_over_original_gap": (
                new_residual / original_gap
                if original_gap > 0 else np.nan
            ),
        })

        np.savez(
            OUT / f"calibrated_MC_{cls}.npz",
            z_enc=calibrated_z,
            logits=calibrated_logits,
            label=labels_mc[mc_rows],
            energy_mev=energy_mc[mc_rows],
            parquet_row=mc_rows,
            classes=np.asarray(classes),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            map_sha256=np.asarray(sha256_file(map_path)),
        )

        pd.DataFrame(summary).to_csv(
            OUT / "efficiency_comparison.csv", index=False
        )

        print(
            f"  raw={raw:.6f}; without E={unconditioned:.6f}; "
            f"with E={conditional:.6f}; TB={tb:.6f}",
            flush=True,
        )

    print("\nEfficiency comparison:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()