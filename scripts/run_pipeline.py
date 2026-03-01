"""Main pipeline execution script.

Usage:
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --config config/config.yaml --stages download,clean,features
    python scripts/run_pipeline.py --config config/config.yaml --stages train --model lstm --stat points

Environment variables:
    ENV=production
    DEVICE=cuda
    BATCH_SIZE=128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.utils.logging import get_logger, setup_logging

logger = get_logger("pipeline")

ALL_STAGES = ["download", "clean", "features", "sequences", "train", "evaluate"]


def run_download(config: Config, force: bool = False) -> None:
    """Run data download stage."""
    from src.data.download import NBADataDownloader
    from src.data.scraper import BasketballReferenceScraper

    data_config = config["data"]
    data_config["data_dir"] = config.get("paths.data_dir", "./data/raw")
    start = data_config["seasons"]["start"]
    end = data_config["seasons"]["end"]

    downloader = NBADataDownloader(data_config)
    downloader.download_all_seasons(start, end, force=force)

    if data_config.get("sources", {}).get("basketball_reference", {}).get("enabled"):
        scraper = BasketballReferenceScraper(data_config)
        scraper.scrape_all_seasons(start, end)


REQUIRED_RAW_COLUMNS = {"PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "PTS", "FG3M", "FTM", "REB"}


def run_clean(config: Config) -> pd.DataFrame:
    """Run data cleaning stage."""
    from src.data.clean import DataCleaner
    from src.utils.io import load_parquet, save_parquet

    data_dir = Path(config.get("paths.data_dir", "./data"))
    raw_dir = data_dir / "raw" / "box_scores"
    processed_dir = data_dir / "processed"

    # Load all season files
    dfs = []
    for path in sorted(raw_dir.glob("player_game_logs_*.parquet")):
        season_df = load_parquet(path)
        missing = REQUIRED_RAW_COLUMNS - set(season_df.columns)
        if missing:
            logger.error(
                "File %s is missing required columns: %s. "
                "This file was likely downloaded with an older version. "
                "Delete it and re-run the download stage, or run with --force-download.",
                path.name,
                missing,
            )
            raise SystemExit(1)
        dfs.append(season_df)

    if not dfs:
        logger.error("No raw data found in %s. Run the download stage first.", raw_dir)
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d raw records across %d files", len(df), len(dfs))

    cleaner = DataCleaner(config["cleaning"])
    df = cleaner.clean_box_scores(df)

    save_parquet(df, processed_dir / "cleaned_box_scores.parquet")
    logger.info("Saved cleaned data to %s", processed_dir / "cleaned_box_scores.parquet")
    return df


def run_features(config: Config, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run feature engineering stage."""
    from src.data.features import FeatureEngineer
    from src.utils.io import load_parquet, save_parquet

    data_dir = Path(config.get("paths.data_dir", "./data"))

    if df is None:
        df = load_parquet(data_dir / "processed" / "cleaned_box_scores.parquet")

    engineer = FeatureEngineer(config["features"])
    df = engineer.build_features(df)

    save_parquet(df, data_dir / "features" / "feature_engineered.parquet")
    logger.info("Saved features to %s", data_dir / "features" / "feature_engineered.parquet")
    return df


def run_sequences(config: Config, df: pd.DataFrame | None = None) -> None:
    """Run sequence formatting stage."""
    from src.data.features import FeatureEngineer
    from src.data.sequences import SequenceBuilder
    from src.utils.io import load_parquet

    data_dir = Path(config.get("paths.data_dir", "./data"))

    if df is None:
        df = load_parquet(data_dir / "features" / "feature_engineered.parquet")

    feature_cols = FeatureEngineer.get_feature_columns(df)
    builder = SequenceBuilder(config["features"])
    sequences = builder.build_sequences(df, feature_cols)
    builder.save_sequences(sequences, str(data_dir / "sequences" / "sequences.npz"))


def run_train(config: Config, model_type: str = "xgboost", target_stat: str = "points") -> None:
    """Run model training stage."""
    from src.data.features import FeatureEngineer
    from src.utils.io import load_parquet

    data_dir = Path(config.get("paths.data_dir", "./data"))
    df = load_parquet(data_dir / "features" / "feature_engineered.parquet")

    feature_cols = FeatureEngineer.get_feature_columns(df)
    target_map = FeatureEngineer.get_target_columns()
    target_col = target_map[target_stat]

    # Temporal split
    splits = config.get("training.splits")
    train_df = df[df["SEASON"] <= splits["train_end_season"]]
    val_df = df[(df["SEASON"] > splits["train_end_season"]) & (df["SEASON"] <= splits["val_end_season"])]

    X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.int64)
    X_val = val_df[feature_cols].fillna(0).values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.int64)

    model_dir = Path(config.get("paths.model_dir", "./outputs/models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "xgboost":
        from src.models.tabular.xgboost_model import XGBoostStatPredictor

        model_config = config["models"]
        model = XGBoostStatPredictor(model_config)
        model.fit((X_train, y_train), (X_val, y_val), config.data)
        model.save(str(model_dir / f"xgboost_{target_stat}.pkl"))

    elif model_type == "logistic":
        from src.models.tabular.logistic import LogisticStatPredictor

        model_config = config["models"]
        model = LogisticStatPredictor(model_config)
        model.fit((X_train, y_train), (X_val, y_val), config.data)
        model.save(str(model_dir / f"logistic_{target_stat}.pkl"))

    elif model_type in ("lstm", "transformer"):
        from src.data.sequences import NBASequenceDataset, SequenceBuilder
        from src.training.trainer import Trainer

        builder = SequenceBuilder(config["features"])

        train_seqs = builder.build_sequences(train_df, feature_cols)
        val_seqs = builder.build_sequences(val_df, feature_cols)

        train_dataset = NBASequenceDataset(train_seqs, target_stat)
        val_dataset = NBASequenceDataset(val_seqs, target_stat)

        seq_config = config["models"]["sequential"]
        model_cfg = {
            **seq_config,
            "num_seq_features": len(feature_cols),
            "num_static_features": 0,
            "max_value": config.get("models.distribution.max_value", 60),
            "target_stats": [target_stat],
        }

        if model_type == "lstm":
            from src.models.sequential.lstm import LSTMStatPredictor
            model = LSTMStatPredictor(model_cfg)
        else:
            from src.models.sequential.transformer import TransformerStatPredictor
            model = TransformerStatPredictor(model_cfg)

        trainer = Trainer({**config["training"], "model_dir": str(model_dir)})
        trainer.train(model, train_dataset, val_dataset, target_stat, f"{model_type}_{target_stat}")

    logger.info("Training complete for %s model on %s.", model_type, target_stat)


def run_evaluate(config: Config, model_type: str = "xgboost", target_stat: str = "points") -> None:
    """Run evaluation stage."""
    from src.data.features import FeatureEngineer
    from src.evaluation.metrics import compute_all_metrics
    from src.utils.io import load_parquet, save_json

    data_dir = Path(config.get("paths.data_dir", "./data"))
    model_dir = Path(config.get("paths.model_dir", "./outputs/models"))
    output_dir = Path(config.get("paths.output_dir", "./outputs"))

    df = load_parquet(data_dir / "features" / "feature_engineered.parquet")
    feature_cols = FeatureEngineer.get_feature_columns(df)
    target_map = FeatureEngineer.get_target_columns()
    target_col = target_map[target_stat]

    splits = config.get("training.splits")
    test_df = df[df["SEASON"] > splits["val_end_season"]]
    X_test = test_df[feature_cols].fillna(0).values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)

    if model_type == "xgboost":
        from src.models.tabular.xgboost_model import XGBoostStatPredictor
        model = XGBoostStatPredictor.load(str(model_dir / f"xgboost_{target_stat}.pkl"))
    elif model_type == "logistic":
        from src.models.tabular.logistic import LogisticStatPredictor
        model = LogisticStatPredictor.load(str(model_dir / f"logistic_{target_stat}.pkl"))
    else:
        logger.error("Sequential model evaluation via CLI not yet supported.")
        return

    pmf = model.predict_pmf(X_test)
    metrics = compute_all_metrics(y_test, pmf)

    logger.info("Evaluation metrics for %s on %s:", model_type, target_stat)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)

    save_json(metrics, output_dir / "reports" / f"metrics_{model_type}_{target_stat}.json")


def main():
    parser = argparse.ArgumentParser(description="Run NBA prop prediction pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(ALL_STAGES),
        help=f"Comma-separated stages to run: {ALL_STAGES}",
    )
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "logistic", "lstm", "transformer"])
    parser.add_argument("--stat", type=str, default="points", choices=["points", "rebounds", "three_pointers_made", "free_throws_made"])
    parser.add_argument("--force-download", action="store_true", help="Force re-download of all data, even if files exist.")
    args = parser.parse_args()

    config = Config.load(args.config, args.override)
    setup_logging(config.get("logging"))

    stages = [s.strip() for s in args.stages.split(",")]
    logger.info("Running pipeline stages: %s", stages)

    df = None

    if "download" in stages:
        logger.info("=== Stage: Download ===")
        run_download(config, force=args.force_download)

    if "clean" in stages:
        logger.info("=== Stage: Clean ===")
        df = run_clean(config)

    if "features" in stages:
        logger.info("=== Stage: Features ===")
        df = run_features(config, df)

    if "sequences" in stages:
        logger.info("=== Stage: Sequences ===")
        run_sequences(config, df)

    if "train" in stages:
        logger.info("=== Stage: Train ===")
        run_train(config, args.model, args.stat)

    if "evaluate" in stages:
        logger.info("=== Stage: Evaluate ===")
        run_evaluate(config, args.model, args.stat)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
