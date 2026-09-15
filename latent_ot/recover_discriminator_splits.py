"""Ricostruisce gli indici del discriminatore con la procedura del Task 2."""

import numpy as np
import train_domain_discriminator as D


def main():
    mc = D.load_latents(D.MC_PATH)
    tb = D.load_latents(D.TB_PATH, is_tb=True)

    if mc["classes"] != tb["classes"]:
        raise ValueError("Classi MC e TB diverse.")
    if mc["checkpoint"] != tb["checkpoint"]:
        raise ValueError("Checkpoint MC e TB diversi.")

    # Numerosità per dominio riportate nell'esecuzione originale.
    expected = {
        "e": (13335, 3339, 7000),
        "p": (18000, 4500, 9000),
        "C": (12000, 3000, 6000),
    }

    recovered = []

    for number, cls in enumerate(D.CLASSES):
        seed = D.SEED + number
        selected, energies, _ = D.select_rows(
            mc, tb, cls, np.random.default_rng(seed)
        )

        for domain in ["mc", "tb"]:
            counts = tuple(
                len(selected[f"{domain}_{part}"])
                for part in ["train", "val", "test"]
            )
            if counts != expected[cls]:
                raise ValueError(
                    f"{cls}, {domain}: conteggi {counts}, "
                    f"attesi {expected[cls]}. Nessun file scritto."
                )

        recovered.append((cls, seed, selected, energies))

    D.OUT.mkdir(parents=True, exist_ok=True)

    for cls, seed, selected, energies in recovered:
        path = D.OUT / f"split_{cls}.npz"

        if path.exists():
            raise FileExistsError(f"Non sovrascrivo: {path}")

        np.savez(
            path,
            **selected,
            common_energies=energies,
            classes=np.asarray(mc["classes"], dtype=str),
            checkpoint_sha256=np.asarray(mc["checkpoint"]),
            seed=seed,
        )

        print(f"{cls}: conteggi coerenti con il Task 2.")
        print(f"Salvato: {path}")


if __name__ == "__main__":
    main()