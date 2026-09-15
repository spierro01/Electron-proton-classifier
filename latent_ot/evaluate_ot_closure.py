"""Task 4: domain-discriminator closure before and after OT.

Reuses Task-2 event splits.
MC=0, TB=1.
Only discriminator training is performed here.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

import train_domain_discriminator as D


HERE = Path(__file__).resolve().parent
TASK2 = HERE / "results/task2_discriminator"
CALIBRATED = HERE / "results/task4_efficiency"
OUT = HERE / "results/task4_closure"


def read_logits(path):
    with np.load(path, allow_pickle=False) as data:
        return data["logits"]


def validate_split(selected, mc, tb, class_index, common):
    """Check independence, class selection and energy matching."""
    for domain, data in [("mc", mc), ("tb", tb)]:
        for split_name in ["train", "val", "test"]:
            rows = selected[f"{domain}_{split_name}"]

            if rows.ndim != 1 or len(rows) == 0:
                raise ValueError("Invalid or empty discriminator split.")
            if len(np.unique(rows)) != len(rows):
                raise ValueError("Duplicate event indices.")
            if np.any(rows < 0) or np.any(rows >= len(data["label"])):
                raise ValueError("Event indices outside dataset.")
            if not np.all(data["label"][rows] == class_index):
                raise ValueError("Wrong particle class in discriminator split.")
            if not np.isin(data["energy"][rows], common).all():
                raise ValueError("Unexpected beam energy.")

        for first, second in [
            ("train", "val"), ("train", "test"), ("val", "test")
        ]:
            if np.intersect1d(
                selected[f"{domain}_{first}"],
                selected[f"{domain}_{second}"],
            ).size:
                raise ValueError("Overlapping discriminator subsets.")

    for name in ["train", "val"]:
        if not tb["is_derivation"][selected[f"tb_{name}"]].all():
            raise ValueError("TB validation used in discriminator fitting.")

    if tb["is_derivation"][selected["tb_test"]].any():
        raise ValueError("TB derivation used in final test.")

    for name in ["train", "val", "test"]:
        for energy in common:
            n_mc = np.sum(
                mc["energy"][selected[f"mc_{name}"]] == energy
            )
            n_tb = np.sum(
                tb["energy"][selected[f"tb_{name}"]] == energy
            )
            if n_mc != n_tb or n_mc == 0:
                raise ValueError("Energy matching differs between domains.")


def make_dataset(
    selected, split_name, representation, calibrated,
    mc, tb, mc_logits, tb_logits, cal, row_lookup
):
    mc_rows = selected[f"mc_{split_name}"]
    tb_rows = selected[f"tb_{split_name}"]

    if calibrated:
        positions = row_lookup[mc_rows]
        if np.any(positions < 0):
            raise ValueError("Missing calibrated MC events.")
        x_mc = cal[representation][positions]
    else:
        x_mc = (
            mc["z"][mc_rows]
            if representation == "z_enc"
            else mc_logits[mc_rows]
        )

    x_tb = (
        tb["z"][tb_rows]
        if representation == "z_enc"
        else tb_logits[tb_rows]
    )

    x = np.concatenate([x_mc, x_tb]).astype(np.float32)
    y = np.concatenate([
        np.zeros(len(x_mc), dtype=np.int64),
        np.ones(len(x_tb), dtype=np.int64),
    ])

    if not np.isfinite(x).all():
        raise ValueError("Non-finite discriminator inputs.")

    return x, y


def main():
    if OUT.exists():
        raise FileExistsError(
            f"{OUT} already exists. Preserve existing results."
        )

    # Check required files before beginning.
    for cls in D.CLASSES:
        for path in [
            TASK2 / f"split_{cls}.npz",
            CALIBRATED / f"calibrated_MC_{cls}.npz",
        ]:
            if not path.is_file():
                raise FileNotFoundError(path)

    # Reuses Task-2 loader, including original TB-mask checks.
    mc = D.load_latents(D.MC_PATH)
    tb = D.load_latents(D.TB_PATH, is_tb=True)

    if mc["classes"] != tb["classes"]:
        raise ValueError("Different class lists.")
    if mc["checkpoint"] != tb["checkpoint"]:
        raise ValueError("Different PID checkpoints.")

    mc_logits = read_logits(D.MC_PATH)
    tb_logits = read_logits(D.TB_PATH)

    OUT.mkdir(parents=True)
    summaries = []
    histories = []

    figures = {}
    for representation in ["z_enc", "logits"]:
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 4.8), layout="constrained"
        )
        figures[representation] = (fig, axes)

    for class_number, cls in enumerate(D.CLASSES):
        class_index = mc["classes"].index(cls)

        with np.load(TASK2 / f"split_{cls}.npz") as data:
            if data["classes"].tolist() != mc["classes"]:
                raise ValueError("Task-2 class list mismatch.")
            if str(data["checkpoint_sha256"].item()) != mc["checkpoint"]:
                raise ValueError("Task-2 checkpoint mismatch.")

            selected = {
                f"{domain}_{part}": data[f"{domain}_{part}"]
                for domain in ["mc", "tb"]
                for part in ["train", "val", "test"]
            }
            common = data["common_energies"]
            seed = int(data["seed"].item())

        validate_split(selected, mc, tb, class_index, common)

        with np.load(
            CALIBRATED / f"calibrated_MC_{cls}.npz",
            allow_pickle=False,
        ) as data:
            if data["classes"].tolist() != mc["classes"]:
                raise ValueError("Calibrated class list mismatch.")
            if str(data["checkpoint_sha256"].item()) != mc["checkpoint"]:
                raise ValueError("Calibrated checkpoint mismatch.")

            cal_rows = data["parquet_row"]
            cal = {
                "z_enc": data["z_enc"],
                "logits": data["logits"],
            }

            if (
                np.any(cal_rows < 0)
                or np.any(cal_rows >= len(mc["label"]))
                or len(np.unique(cal_rows)) != len(cal_rows)
            ):
                raise ValueError("Invalid calibrated row indices.")

            np.testing.assert_array_equal(
                data["label"], mc["label"][cal_rows]
            )
            np.testing.assert_array_equal(
                data["energy_mev"], mc["energy"][cal_rows]
            )
            if not np.all(data["label"] == class_index):
                raise ValueError("Wrong calibrated class.")

        row_lookup = np.full(len(mc["label"]), -1, dtype=np.int64)
        row_lookup[cal_rows] = np.arange(len(cal_rows))

        for representation in ["z_enc", "logits"]:
            for status, calibrated in [
                ("raw", False), ("calibrated", True)
            ]:
                tag = f"{cls}_{representation}_{status}"
                print(f"\n{tag}", flush=True)

                x_train, y_train = make_dataset(
                    selected, "train", representation, calibrated,
                    mc, tb, mc_logits, tb_logits, cal, row_lookup
                )
                x_val, y_val = make_dataset(
                    selected, "val", representation, calibrated,
                    mc, tb, mc_logits, tb_logits, cal, row_lookup
                )

                # Identical training permutation before and after OT.
                order = np.random.default_rng(seed).permutation(len(y_train))
                x_train, y_train = x_train[order], y_train[order]

                candidate = D.new_model(seed, warm_start=True)
                best_auc = -np.inf
                best_iterations = None

                # Model selection uses internal validation only.
                for iterations in D.ITERATION_CANDIDATES:
                    candidate.set_params(max_iter=iterations)
                    candidate.fit(x_train, y_train)
                    auc = float(roc_auc_score(
                        y_val, candidate.predict_proba(x_val)[:, 1]
                    ))

                    histories.append({
                        "cls": cls,
                        "representation": representation,
                        "status": status,
                        "iterations": iterations,
                        "internal_validation_auc": auc,
                    })

                    print(
                        f"  iterations={iterations}: "
                        f"internal AUC={auc:.6f}",
                        flush=True,
                    )

                    if auc > best_auc:
                        best_auc = auc
                        best_iterations = iterations

                model = D.new_model(seed, iterations=best_iterations)
                model.fit(x_train, y_train)

                # Final test is evaluated only after model selection.
                x_test, y_test = make_dataset(
                    selected, "test", representation, calibrated,
                    mc, tb, mc_logits, tb_logits, cal, row_lookup
                )
                scores = model.predict_proba(x_test)[:, 1]
                final_auc = float(roc_auc_score(y_test, scores))
                fpr, tpr, thresholds = roc_curve(y_test, scores)

                joblib.dump(model, OUT / f"{tag}.joblib")
                np.savez(
                    OUT / f"{tag}_evaluation.npz",
                    domain_label=y_test,
                    score_tb=scores,
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=thresholds,
                    mc_test_rows=selected["mc_test"],
                    tb_test_rows=selected["tb_test"],
                )

                summaries.append({
                    "cls": cls,
                    "representation": representation,
                    "status": status,
                    "selected_iterations": best_iterations,
                    "internal_validation_auc": best_auc,
                    "heldout_auc": final_auc,
                    "n_test_MC": len(selected["mc_test"]),
                    "n_test_TB": len(selected["tb_test"]),
                })

                pd.DataFrame(summaries).to_csv(
                    OUT / "auc_summary.csv", index=False
                )
                pd.DataFrame(histories).to_csv(
                    OUT / "training_history.csv", index=False
                )

                ax = figures[representation][1][class_number]
                ax.plot(
                    fpr, tpr,
                    label=f"{status}: AUC={final_auc:.4f}"
                )
                print(f"  FINAL TEST AUC={final_auc:.6f}", flush=True)

                del x_train, x_val, x_test, candidate, model

        del cal, row_lookup

    for representation, (fig, axes) in figures.items():
        for ax, cls in zip(axes, D.CLASSES):
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_title(cls)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right")

        fig.suptitle(
            f"MC vs TB validation — {representation}\n"
            "Before and after OT; matched energy mixtures"
        )
        for extension in ["png", "pdf"]:
            fig.savefig(
                OUT / f"roc_{representation}.{extension}",
                dpi=200,
            )
        plt.close(fig)

    table = pd.DataFrame(summaries).pivot(
        index=["cls", "representation"],
        columns="status",
        values="heldout_auc",
    ).reset_index()
    table["auc_raw_minus_calibrated"] = table["raw"] - table["calibrated"]
    table.to_csv(OUT / "auc_comparison.csv", index=False)

    config = {
        "checkpoint_sha256": mc["checkpoint"],
        "split_source": str(TASK2),
        "iteration_candidates": D.ITERATION_CANDIDATES,
        "domain_labels": {"MC": 0, "TB": 1},
        "TB_final_test": "Original validation half only.",
        "note": (
            "Test events are held out from discriminator fitting. "
            "MC events were available during OT fitting; TB validation "
            "was excluded from OT fitting. Energy mixtures are matched "
            "for discriminator evaluation, whereas integrated PID "
            "efficiencies use the original sample mixtures."
        ),
    }
    (OUT / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nAUC comparison:")
    print(table.to_string(index=False))
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()