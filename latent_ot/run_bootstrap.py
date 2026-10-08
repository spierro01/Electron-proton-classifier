"""Sequential bootstrap training/evaluation. Run inside tmux; finished maps are reused."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import fcntl

import numpy as np
import torch
from train_conditional_ot_bootstrap import read_class, statistics
from train_ot import sha256_file

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_conditional_ot_bootstrap.py"
EVALUATE = HERE / "evaluate_bootstrap.py"
KEYS = ["architecture", "batch_size", "outer_steps", "identity_steps", "g_updates", "f_updates",
        "lr", "min_lr", "betas", "weight_decay", "grad_clip", "seed", "common_energies",
        "classes", "class_name", "checkpoint_sha256", "tb_split_sha256"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_draw(folder):
    config = read_json(folder / "config.json")
    mask_path = HERE / "derived/tb_is_derivation.npy"
    assert sha256_file(mask_path) == config["tb_split_sha256"]
    mask = np.load(mask_path, allow_pickle=False)
    target, energy, rows, _, _ = read_class(HERE / "derived/latents_TB.npz", config["class_name"], mask=mask)
    keep = np.isin(energy, config["common_energies"])
    target, energy, rows = target[keep], energy[keep], rows[keep]
    draw = np.random.default_rng(config["bootstrap_seed"]).choice(len(rows), size=len(rows), replace=True)
    with np.load(folder / "selection.npz", allow_pickle=False) as selection:
        np.testing.assert_array_equal(selection["tb_original_derivation_rows"], rows)
        np.testing.assert_array_equal(selection["bootstrap_indices"], draw)
        np.testing.assert_array_equal(selection["tb_derivation_rows"], rows[draw])
        np.testing.assert_array_equal(selection["tb_energy"], energy[draw])
        assert mask[selection["tb_derivation_rows"]].all()
    saved = torch.load(folder / "map.pt", map_location="cpu", weights_only=True)
    assert saved["metadata"] == config
    mean, std = statistics(target[draw])
    np.testing.assert_array_equal(np.asarray(saved["stats"]["target_mean"], dtype=np.float32), mean)
    np.testing.assert_array_equal(np.asarray(saved["stats"]["target_std"], dtype=np.float32), std)
    assert config["TB_validation_used"] is False
    assert read_json(folder / "verification.json")["reload_invariance_passed"] is True
    print("PASS: reproducible bootstrap, derivation only, target statistics recomputed, reload invariance.", flush=True)


def complete(folder):
    if not (folder / "map.pt").is_file() or not (folder / "verification.json").is_file():
        return False
    check = read_json(folder / "verification.json")
    return check.get("reload_invariance_passed") is True and check.get("TB_validation_used") is False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="Restart interrupted training from zero in a new attempt directory.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.replicas < 2:
        raise ValueError("At least two replicas required for sample standard deviation.")
    root = HERE / "results" / ("bootstrap_smoke" if args.smoke else "bootstrap_mod4")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another bootstrap runner is already active in this directory.")
    manifest = []
    fingerprints = {p.name: sha256_file(p) for p in [HERE / "derived/latents_MC.npz",
                    HERE / "derived/latents_TB.npz", HERE / "derived/tb_is_derivation.npy"]}
    implementation = {p.name: sha256_file(p) for p in [TRAIN, HERE / "conditional_ot_model.py", HERE / "train_ot.py"]}
    for class_index, cls in enumerate(["e"] if args.smoke else ["e", "p", "C"]):
        nominal_folder = HERE / f"results/task6_spark_{cls}_mod4"
        nominal = read_json(nominal_folder / "config.json")
        if not complete(nominal_folder):
            raise ValueError("Incomplete nominal map: " + str(nominal_folder))
        arch = nominal["architecture"]
        if arch != {"in_dim": 64, "hidden_dim": 2048, "num_layers": 4,
                    "condition_dim": 32, "input_modulation": True}:
            raise ValueError("Expected nominal mod4 architecture.")
        constants = {"g_updates": 10, "f_updates": 4, "lr": 5e-4, "min_lr": 5e-6,
                     "betas": [0., .9], "weight_decay": 0., "grad_clip": 10.,
                     "batch_size": 1024, "outer_steps": 10000, "identity_steps": 1000}
        for key, value in constants.items():
            if nominal[key] != value:
                raise ValueError("Unexpected nominal parameter: " + key)
        for replica in range(1, (1 if args.smoke else args.replicas) + 1):
            bootstrap_seed = 1000 + class_index * 10000 + replica
            job = root / cls / f"replica_{replica:03d}"
            expected = {key: nominal[key] for key in KEYS}
            if args.smoke:
                expected.update(outer_steps=20, identity_steps=10)
            expected.update(bootstrap_seed=bootstrap_seed, input_sha256=fingerprints,
                            implementation_sha256=implementation)
            attempts = sorted(job.glob("attempt_*"))
            finished = [folder for folder in attempts if complete(folder / "training")]
            if len(finished) > 1:
                raise ValueError("Multiple completed attempts; inspect " + str(job))
            if finished:
                attempt = finished[0]
                print("Reusing completed training:", attempt, flush=True)
            else:
                if attempts and not args.retry_incomplete:
                    raise RuntimeError(f"Interrupted job: {job}. Check no process is running; then use --retry-incomplete to restart it from zero.")
                attempt = job / f"attempt_{len(attempts) + 1:03d}"
                command = [sys.executable, "-u", str(TRAIN), "--class-name", cls,
                           "--bootstrap-seed", str(bootstrap_seed), "--seed", str(nominal["seed"]),
                           "--hidden", "2048", "--layers", "4", "--input-modulation",
                           "--batch-size", "1024", "--steps", str(expected["outer_steps"]),
                           "--identity-steps", str(expected["identity_steps"]),
                           "--log-every", "1" if args.smoke else "100", "--device", args.device,
                           "--out", str(attempt / "training")]
                print(f"\nBOOTSTRAP {cls}, replica {replica}, seed {bootstrap_seed}", flush=True)
                subprocess.run(command, cwd=HERE, check=True)
            folder = attempt / "training"
            config = read_json(folder / "config.json")
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(f"Saved training mismatch: {folder}, {key}")
            audit_draw(folder)
            if args.smoke:
                print("PASS: BOOTSTRAP SMOKE TEST. No scientific result from this short run.", flush=True)
                return
            evaluations = sorted(attempt.glob("evaluation_*"))
            ready = [ev for ev in evaluations if (ev / "evaluation_complete.json").is_file()]
            if len(ready) > 1:
                raise ValueError("Duplicate evaluations: " + str(attempt))
            if ready:
                evaluation = ready[0]
            else:
                evaluation = attempt / f"evaluation_{len(evaluations) + 1:03d}"
                subprocess.run([sys.executable, "-u", str(EVALUATE), "--class-name", cls,
                                "--map-dir", str(folder), "--out", str(evaluation)], cwd=HERE, check=True)
            check = read_json(evaluation / "evaluation_complete.json")
            if check["map_sha256"] != sha256_file(folder / "map.pt") or check["bootstrap_seed"] != bootstrap_seed:
                raise ValueError("Stale bootstrap evaluation.")
            manifest.append({"cls": cls, "replica": replica, "bootstrap_seed": bootstrap_seed,
                             "evaluation": str(evaluation.resolve()), "training": str(folder.resolve())})
            root.mkdir(parents=True, exist_ok=True)
            pending = root / "manifest.pending.json"
            pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            pending.replace(root / "manifest.json")
    print("COMPLETED: all requested bootstrap maps and evaluations.", flush=True)
    subprocess.run([sys.executable, "-u", str(HERE / "summarize_bootstrap.py"),
                    "--manifest", str(root / "manifest.json")], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
