"""Report generation script.

Usage:
    python scripts/generate_report.py --config config/config.yaml
    python scripts/generate_report.py --config config/config.yaml --models xgboost,logistic --stat points
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.reporting.pdf_report import ReportGenerator
from src.utils.io import load_json
from src.utils.logging import get_logger, setup_logging

logger = get_logger("report")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--models", type=str, default="xgboost,logistic")
    parser.add_argument("--stat", type=str, default="points")
    parser.add_argument("--output", type=str, default="evaluation_report.pdf")
    args = parser.parse_args()

    config = Config.load(args.config, args.override)
    setup_logging(config.get("logging"))

    output_dir = Path(config.get("paths.output_dir", "./outputs"))
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"

    # Load metrics for each model
    model_names = [m.strip() for m in args.models.split(",")]
    all_metrics = {}
    for model_name in model_names:
        metrics_path = reports_dir / f"metrics_{model_name}_{args.stat}.json"
        if metrics_path.exists():
            all_metrics[model_name] = load_json(str(metrics_path))
        else:
            logger.warning("Metrics not found for %s: %s", model_name, metrics_path)

    if not all_metrics:
        logger.error("No metrics found. Run evaluation first.")
        return

    # Collect figure paths
    figures = {}
    for fig_path in figures_dir.glob("*.png"):
        figures[fig_path.stem] = str(fig_path)

    # Build report data
    report_data = {
        "executive_summary": (
            f"Evaluation results for NBA player prop prediction models "
            f"targeting {args.stat}. Models evaluated: {', '.join(model_names)}."
        ),
        "methodology": (
            "Models predict probability distributions over player counting stats "
            "using historical game data from 2015-present. Features are constructed "
            "with strict T-30 minute information cutoff to prevent data leakage. "
            "Evaluation uses temporal train/validation/test splits by season."
        ),
        "metrics": all_metrics,
        "figures": figures,
    }

    # Generate report
    report_config = config.get("reporting", {})
    report_config["output_dir"] = str(reports_dir)
    generator = ReportGenerator(report_config)
    report_path = generator.generate(report_data, args.output)
    logger.info("Report generated: %s", report_path)


if __name__ == "__main__":
    main()
