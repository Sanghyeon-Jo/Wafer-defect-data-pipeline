"""Helper script to train (or refresh) the wafer defect risk model."""

from pathlib import Path

from ml_pipeline import ensure_artifacts


def main() -> None:
    artifacts_dir = Path("artifacts")
    artifacts, metrics, _ = ensure_artifacts(artifacts_dir)

    if metrics:
        print("Trained RandomForestRegressor.")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print("Artifacts already present; skipped training.")
    print(f"Artifacts stored in: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()


