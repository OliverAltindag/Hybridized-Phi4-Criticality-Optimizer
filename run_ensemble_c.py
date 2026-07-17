import argparse
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOW_LAMBDA_TARGET_SLOPE = 0.020609922762014173
LOW_LAMBDA_TARGET_INTERCEPT = 0.9916447231591535
LOG10_TARGET_SLOPE = 0.0009153328160559637
LOG10_TARGET_INTERCEPT = 0.9937789902721982

GAMMA_SCALE_CHOICES = ("none", "lambda", "sqrt-lambda")
INIT_CHOICES = ("hot", "normal", "uniform", "ones", "signs")
TEFF_TARGET_CHOICES = ("one", "low-lambda-fit", "log10-fit")


def _parse_lambdas(text):
    return [float(part) for part in text.split(",") if part.strip()]


def _teff_target(lambda_L, teff_target_mode):
    if teff_target_mode == "one":
        return 1.0
    if teff_target_mode == "low-lambda-fit":
        if lambda_L >= 0.25:
            raise ValueError(
                "low-lambda-fit target is only valid for lambda_L < 0.25"
            )
        return LOW_LAMBDA_TARGET_SLOPE * lambda_L + LOW_LAMBDA_TARGET_INTERCEPT
    if teff_target_mode == "log10-fit":
        if lambda_L <= 0.0:
            raise ValueError("log10-fit target requires lambda_L > 0")
        return LOG10_TARGET_INTERCEPT + LOG10_TARGET_SLOPE * math.log10(lambda_L)

    raise ValueError(f"unknown T_eff target mode: {teff_target_mode}")


def _effective_gamma(base_gamma, lambda_L, gamma_scale):
    if gamma_scale == "none":
        return base_gamma
    if gamma_scale == "lambda":
        return base_gamma * lambda_L
    if gamma_scale == "sqrt-lambda":
        if lambda_L < 0.0:
            raise ValueError("sqrt-lambda gamma scaling requires nonnegative lambda")
        return base_gamma * float(np.sqrt(lambda_L))

    raise ValueError(f"unknown gamma scale mode: {gamma_scale}")


def _seed_state(base_seed, lambda_index, replica):
    sequence = np.random.SeedSequence([base_seed, lambda_index, replica])
    state = sequence.generate_state(4, dtype=np.uint64)
    if not np.any(state):
        state[0] = np.uint64(1)
    return state


def _initial_lattice(N, base_seed, lambda_index, replica, mode):
    sequence = np.random.SeedSequence([base_seed, lambda_index, replica, 99991])
    rng = np.random.default_rng(sequence)

    if mode == "hot":
        return rng.uniform(-0.5, 0.5, size=(N, N))
    if mode == "normal":
        return rng.normal(size=(N, N))
    if mode == "uniform":
        return rng.uniform(-1.0, 1.0, size=(N, N))
    if mode == "ones":
        return np.ones((N, N), dtype=float)
    if mode == "signs":
        return rng.choice(np.array([-1.0, 1.0]), size=(N, N))

    raise ValueError(f"unknown lattice init mode: {mode}")


def _require_c_extension():
    spec = importlib.util.find_spec("invaded_phi4_c")
    if spec is None:
        raise SystemExit(
            "Could not import the compiled invaded_phi4_c extension.\n"
            f"Expected to find it from: {SCRIPT_DIR}\n\n"
            "Build it first with the same Python interpreter used to run this script:\n"
            "  python -m pip install -r requirements.txt\n"
            "  python setup.py build_ext --inplace\n"
            "Then confirm a file like invaded_phi4_c*.pyd or invaded_phi4_c*.so "
            "exists in this folder."
        )


def _batch_means_error(values, burn_in, block_size):
    data = np.asarray(values, dtype=float)[burn_in:]
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("no finite mu_history samples after burn-in")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    mean = float(np.mean(data))
    block_count = data.size // block_size
    if block_count < 2:
        return {
            "mean": mean,
            "stderr": float("nan"),
            "block_size": int(block_size),
            "block_count": int(block_count),
            "sample_count": int(data.size),
        }

    trimmed = data[:block_count * block_size]
    block_means = trimmed.reshape(block_count, block_size).mean(axis=1)
    return {
        "mean": mean,
        "stderr": float(np.std(block_means, ddof=1) / np.sqrt(block_count)),
        "block_size": int(block_size),
        "block_count": int(block_count),
        "sample_count": int(data.size),
    }


def _invaded_cluster_phi4_with_target(
    lattice,
    N,
    state,
    total_steps,
    lambda_L,
    mu_sq_init,
    gamma,
    teff_target,
):
    from invaded_phi4_c import metropolis_phi4, swedson_wang_phi4

    mu_sq = mu_sq_init
    mu_history = []
    teff_history = []
    progress_interval = max(1, total_steps // 10)

    for step in range(total_steps):
        lattice, T_eff = swedson_wang_phi4(lattice, N, state)

        deviation = T_eff - teff_target
        deviation = max(min(deviation, 2.0), -2.0)
        mu_sq = mu_sq + gamma * deviation

        lattice = metropolis_phi4(lattice, N, state, sweeps=5, lambda_L=lambda_L, mu_sq=mu_sq)

        mu_history.append(mu_sq)
        teff_history.append(T_eff)

        if step > 0 and step % progress_interval == 0:
            print(
                f"Step {step}: T_eff = {T_eff:.4f}, "
                f"target = {teff_target:.4f}, mu_sq = {mu_sq:.4f}"
            )

    return lattice, mu_history, teff_history


def _run_one(config):
    from invaded_phi4_c import invaded_cluster_phi4

    lambda_L = config["lambda_L"]
    lambda_index = config["lambda_index"]
    replica = config["replica"]
    N = config["N"]

    state = _seed_state(config["seed"], lambda_index, replica)
    lattice = np.ascontiguousarray(_initial_lattice(
        N,
        config["seed"],
        lambda_index,
        replica,
        config["init"],
    ), dtype=float)

    if config["teff_target_mode"] == "one":
        lattice, mu_history, teff_history = invaded_cluster_phi4(
            lattice,
            N,
            state,
            config["total_steps"],
            lambda_L,
            config["mu_sq_init"],
            config["gamma"],
        )
    else:
        lattice, mu_history, teff_history = _invaded_cluster_phi4_with_target(
            lattice,
            N,
            state,
            config["total_steps"],
            lambda_L,
            config["mu_sq_init"],
            config["gamma"],
            config["teff_target"],
        )

    mu_history = np.asarray(mu_history, dtype=float)
    teff_history = np.asarray(teff_history, dtype=float)
    finite_teff = np.isfinite(teff_history)
    summary = _batch_means_error(
        mu_history,
        burn_in=config["burn_in"],
        block_size=config["block_size"],
    )

    result = {
        "lambda_L": float(lambda_L),
        "lambda_index": int(lambda_index),
        "replica": int(replica),
        "seed": int(config["seed"]),
        "N": int(N),
        "total_steps": int(config["total_steps"]),
        "burn_in": int(config["burn_in"]),
        "block_size": int(config["block_size"]),
        "gamma": float(config["gamma"]),
        "gamma_base": float(config["gamma_base"]),
        "gamma_scale": config["gamma_scale"],
        "teff_target": float(config["teff_target"]),
        "teff_target_mode": config["teff_target_mode"],
        "mu_sq_init": float(config["mu_sq_init"]),
        "init": config["init"],
        "mu_mean": summary["mean"],
        "mu_stderr_batch": summary["stderr"],
        "mu_block_count": summary["block_count"],
        "mu_sample_count": summary["sample_count"],
        "teff_finite_fraction": float(np.mean(finite_teff)),
        "final_mu": float(mu_history[-1]),
        "final_teff": float(teff_history[-1]),
        "final_rng_state": state.tolist(),
    }

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lambda_{lambda_index:04d}_replica_{replica:04d}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        mu_history=mu_history,
        teff_history=teff_history,
        final_lattice=lattice,
        **result,
    )
    with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    return result


def _validate_args(args):
    if args.N <= 0:
        raise SystemExit("--N must be positive")
    if args.replicas < 1:
        raise SystemExit("--replicas must be at least 1")
    if args.total_steps <= 0:
        raise SystemExit("--total-steps must be positive")
    if args.burn_in < 0:
        raise SystemExit("--burn-in must be nonnegative")
    if args.burn_in >= args.total_steps:
        raise SystemExit("--burn-in must be smaller than --total-steps")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not args.lambdas:
        raise SystemExit("--lambdas must contain at least one value")


def _summarize(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["lambda_index"], []).append(result)

    summaries = []
    for lambda_index, group in sorted(grouped.items()):
        mu_values = np.array([item["mu_mean"] for item in group], dtype=float)
        lambda_L = group[0]["lambda_L"]
        if mu_values.size > 1:
            ensemble_stderr = float(np.std(mu_values, ddof=1) / np.sqrt(mu_values.size))
        else:
            ensemble_stderr = float("nan")

        summaries.append({
            "lambda_L": float(lambda_L),
            "lambda_index": int(lambda_index),
            "replicas": int(mu_values.size),
            "gamma": float(group[0]["gamma"]),
            "gamma_base": float(group[0]["gamma_base"]),
            "gamma_scale": group[0]["gamma_scale"],
            "teff_target": float(group[0]["teff_target"]),
            "teff_target_mode": group[0]["teff_target_mode"],
            "mu_mean": float(np.mean(mu_values)),
            "mu_stderr_ensemble": ensemble_stderr,
            "mu_replica_std": float(np.std(mu_values, ddof=1)) if mu_values.size > 1 else float("nan"),
            "mean_teff_finite_fraction": float(np.mean([item["teff_finite_fraction"] for item in group])),
        })

    return summaries


def _group_configs_by_lambda(configs):
    grouped = {}
    for config in configs:
        grouped.setdefault(config["lambda_index"], []).append(config)
    return grouped


def _run_configs(configs, workers_per_lambda, serial_lambdas):
    if serial_lambdas or len(_group_configs_by_lambda(configs)) == 1:
        if workers_per_lambda == 1:
            return [_run_one(config) for config in configs]

        results = []
        with ProcessPoolExecutor(max_workers=workers_per_lambda) as executor:
            futures = [executor.submit(_run_one, config) for config in configs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"finished lambda_index={result['lambda_index']} "
                    f"replica={result['replica']} mu={result['mu_mean']:.8g}"
                )
        return results

    grouped = _group_configs_by_lambda(configs)
    lambda_count = len(grouped)
    total_workers = workers_per_lambda * lambda_count
    print(
        f"running {lambda_count} lambdas in parallel with "
        f"{workers_per_lambda} workers per lambda "
        f"({total_workers} worker processes total)"
    )

    results = []
    executors = []
    futures = []
    try:
        for lambda_index, group in sorted(grouped.items()):
            executor = ProcessPoolExecutor(max_workers=workers_per_lambda)
            executors.append(executor)
            for config in group:
                futures.append(executor.submit(_run_one, config))
            print(
                f"submitted lambda_index={lambda_index} "
                f"replicas={len(group)} workers={workers_per_lambda}"
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"finished lambda_index={result['lambda_index']} "
                f"replica={result['replica']} mu={result['mu_mean']:.8g}"
            )
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

    return results


def _plot_convergence(result, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = f"lambda_{result['lambda_index']:04d}_replica_{result['replica']:04d}"
    data = np.load(out_dir / f"{stem}.npz")
    mu_history = data["mu_history"]
    teff_history = data["teff_history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(mu_history, color="blue", alpha=0.8)
    ax1.axvline(result["burn_in"], color="black", linestyle="--", linewidth=1)
    ax1.set_title(r"Convergence of $\mu^2$")
    ax1.set_xlabel("Macro-Sweeps")
    ax1.set_ylabel(r"$\mu^2$")
    ax1.grid(True)

    ax2.plot(teff_history, color="red", alpha=0.6)
    teff_target = float(result.get("teff_target", 1.0))
    ax2.axhline(teff_target, color="black", linestyle="--", linewidth=2,
                label=fr"Target T={teff_target:.6g}")
    ax2.axvline(result["burn_in"], color="black", linestyle="--", linewidth=1)
    ax2.set_title("Effective Temperature at Percolation")
    ax2.set_xlabel("Macro-Sweeps")
    ax2.set_ylabel(r"$T_{eff}$")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_convergence.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run independent C-backed phi4 invaded-cluster replicas for "
            "ensemble error estimates."
        )
    )
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--lambdas", type=_parse_lambdas, required=True,
                        help="Comma-separated lambda_L values, for example: 0.5,1.0,1.5")
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--burn-in", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--gamma-scale", choices=GAMMA_SCALE_CHOICES, default="none",
                        help="Scale the per-lambda feedback rate: none uses gamma directly; "
                             "lambda uses gamma*lambda_L; sqrt-lambda uses gamma*sqrt(lambda_L).")
    parser.add_argument("--teff-target", choices=TEFF_TARGET_CHOICES, default="one",
                        help="Feedback target for T_eff. 'one' preserves the original algorithm. "
                             "'low-lambda-fit' uses T_eff = 0.020609922762014173*lambda + "
                             "0.9916447231591535 and is only valid for lambda < 0.25. "
                             "'log10-fit' uses T_eff = 0.9937789902721982 + "
                             "0.0009153328160559637*log10(lambda).")
    parser.add_argument("--mu-sq-init", type=float, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes per lambda. Total workers are workers times active lambdas.")
    parser.add_argument("--out-dir", default="ensemble_results_c")
    parser.add_argument("--init", choices=INIT_CHOICES, default="hot")
    parser.add_argument("--lambda-index", type=int, default=None,
                        help="Run only this zero-based lambda index, useful for cluster arrays.")
    parser.add_argument("--replica-start", type=int, default=0,
                        help="First replica id to run, useful for cluster arrays.")
    parser.add_argument("--plot", action="store_true",
                        help="Save convergence PNGs for completed replicas.")
    parser.add_argument("--plot-limit", type=int, default=4,
                        help="Maximum number of convergence plots to save when --plot is used.")
    parser.add_argument("--serial-lambdas", action="store_true",
                        help="Run lambda groups one after another instead of concurrently.")
    args = parser.parse_args()
    _require_c_extension()
    _validate_args(args)

    lambdas = args.lambdas
    lambda_items = list(enumerate(lambdas))
    if args.lambda_index is not None:
        lambda_items = [item for item in lambda_items if item[0] == args.lambda_index]
        if not lambda_items:
            raise SystemExit(f"lambda index {args.lambda_index} is out of range")

    configs = []
    for lambda_index, lambda_L in lambda_items:
        effective_gamma = _effective_gamma(args.gamma, lambda_L, args.gamma_scale)
        teff_target = _teff_target(lambda_L, args.teff_target)
        for offset in range(args.replicas):
            replica = args.replica_start + offset
            configs.append({
                "N": args.N,
                "lambda_L": lambda_L,
                "lambda_index": lambda_index,
                "replica": replica,
                "total_steps": args.total_steps,
                "burn_in": args.burn_in,
                "block_size": args.block_size,
                "gamma": effective_gamma,
                "gamma_base": args.gamma,
                "gamma_scale": args.gamma_scale,
                "teff_target": teff_target,
                "teff_target_mode": args.teff_target,
                "mu_sq_init": args.mu_sq_init,
                "seed": args.seed,
                "out_dir": args.out_dir,
                "init": args.init,
            })

    results = _run_configs(
        configs,
        workers_per_lambda=args.workers,
        serial_lambdas=args.serial_lambdas,
    )

    summaries = _summarize(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)

    if args.plot:
        for result in sorted(results, key=lambda item: (item["lambda_index"], item["replica"]))[:args.plot_limit]:
            _plot_convergence(result, out_dir)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
