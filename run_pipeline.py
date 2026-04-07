"""
Master pipeline script — trains agents, evaluates all strategies, and generates plots.

Usage
-----
  # Full run (train + evaluate + plot):
  python run_pipeline.py

  # Skip training, use existing models:
  python run_pipeline.py --eval-only

  # Force retrain even if models exist:
  python run_pipeline.py --retrain

  # Quick smoke-test with fewer episodes:
  python run_pipeline.py --eval-only --n-episodes 50

Outputs
-------
  models/ppo_hedger.zip             trained PPO agent
  models/sac_hedger.zip             trained SAC agent
  results/evaluation_results.csv   all metrics in a tidy table
  results/plots/*.png               6 analysis figures
  results/tensorboard/              TensorBoard training logs
"""

import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _run(script: str, label: str):
    """Run a Python script in a subprocess and stream its output."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, script)],
        cwd=ROOT,
        check=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  Done in {elapsed:.1f}s")


def _models_exist() -> bool:
    return (
        os.path.exists(os.path.join(ROOT, "models", "ppo_hedger.zip"))
        and os.path.exists(os.path.join(ROOT, "models", "sac_hedger.zip"))
    )


def _patch_n_episodes(n: int):
    """Temporarily patch N_EPISODES in full_evaluation.py for quick runs."""
    path = os.path.join(ROOT, "evaluation", "full_evaluation.py")
    with open(path) as f:
        src = f.read()
    patched = src.replace(
        "N_EPISODES = 500",
        f"N_EPISODES = {n}",
    )
    with open(path, "w") as f:
        f.write(patched)
    return src   # return original for restoration


def _restore_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="RL Derivative Hedging pipeline")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain agents even if saved models exist",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; evaluate existing models",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=500,
        help="Episodes per strategy/scenario for evaluation (default: 500)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  RL DERIVATIVE HEDGING — FULL PIPELINE")
    print("="*70)

    # ------------------------------------------------------------------
    # Step 1: Training
    # ------------------------------------------------------------------
    if args.eval_only:
        print("\n[--eval-only] Skipping training.")
        if not _models_exist():
            print(
                "\nWARNING: No trained models found at models/ppo_hedger.zip "
                "or models/sac_hedger.zip.\n"
                "Evaluation will run without RL agents.  "
                "Run without --eval-only to train first."
            )
    else:
        if _models_exist() and not args.retrain:
            print(
                "\n[INFO] Trained models already exist. "
                "Pass --retrain to force retraining."
            )
        else:
            _run("training/train_ppo.py", "Step 1a — Training PPO agent")
            _run("training/train_sac.py", "Step 1b — Training SAC agent")

    # ------------------------------------------------------------------
    # Step 2: Evaluation
    # ------------------------------------------------------------------
    original_src = None
    eval_path = os.path.join(ROOT, "evaluation", "full_evaluation.py")

    if args.n_episodes != 500:
        print(f"\n[INFO] Using {args.n_episodes} episodes per strategy/scenario.")
        original_src = _patch_n_episodes(args.n_episodes)

    try:
        _run("evaluation/full_evaluation.py", "Step 2 — Running full evaluation")
    finally:
        if original_src is not None:
            _restore_file(eval_path, original_src)

    # ------------------------------------------------------------------
    # Step 3: Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        _run("evaluation/visualize_results.py", "Step 3 — Generating plots")
    else:
        print("\n[--no-plots] Skipping plot generation.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print("\nKey outputs:")
    for path in [
        "models/ppo_hedger.zip",
        "models/sac_hedger.zip",
        "results/evaluation_results.csv",
        "results/plots/01_pnl_distributions.png",
        "results/plots/02_risk_return.png",
        "results/plots/03_sharpe_comparison.png",
        "results/plots/04_var_cvar_comparison.png",
        "results/plots/05_transaction_costs.png",
        "results/plots/06_learning_curves.png",
    ]:
        full = os.path.join(ROOT, path)
        status = "✓" if os.path.exists(full) else "✗ not generated"
        print(f"  {status}  {path}")

    print(
        "\nTo view results:\n"
        "  cat results/evaluation_results.csv\n"
        "  open results/plots/03_sharpe_comparison.png\n"
    )


if __name__ == "__main__":
    main()
