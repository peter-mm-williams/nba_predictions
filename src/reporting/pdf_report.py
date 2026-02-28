"""PDF report generation for model evaluation results."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger("reporting.pdf")

DEFAULT_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<style>
    body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
    h1 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }
    h2 { color: #2e86c1; margin-top: 30px; }
    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #2e86c1; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .metric-value { font-weight: bold; font-size: 1.1em; }
    .section { page-break-inside: avoid; margin-bottom: 30px; }
    img { max-width: 100%; margin: 10px 0; }
    .summary-box { background: #eaf2f8; padding: 15px; border-radius: 5px; margin: 15px 0; }
</style>
</head>
<body>
    <h1>NBA Player Prop Prediction Report</h1>

    {% if executive_summary %}
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>{{ executive_summary }}</p>
        </div>
    </div>
    {% endif %}

    {% if methodology %}
    <div class="section">
        <h2>Methodology</h2>
        <p>{{ methodology }}</p>
    </div>
    {% endif %}

    {% if metrics %}
    <div class="section">
        <h2>Model Performance</h2>
        <table>
            <tr>
                <th>Model</th>
                {% for metric_name in metric_names %}
                <th>{{ metric_name }}</th>
                {% endfor %}
            </tr>
            {% for model_name, model_metrics in metrics.items() %}
            <tr>
                <td>{{ model_name }}</td>
                {% for metric_name in metric_names %}
                <td class="metric-value">{{ "%.4f"|format(model_metrics[metric_name]) }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    {% if feature_importance %}
    <div class="section">
        <h2>Feature Importance</h2>
        <table>
            <tr><th>Feature</th><th>Importance</th></tr>
            {% for name, value in feature_importance[:20] %}
            <tr><td>{{ name }}</td><td>{{ "%.4f"|format(value) }}</td></tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    {% if figures %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for fig_name, fig_path in figures.items() %}
        <h3>{{ fig_name }}</h3>
        <img src="{{ fig_path }}" alt="{{ fig_name }}">
        {% endfor %}
    </div>
    {% endif %}

    {% if calibration_analysis %}
    <div class="section">
        <h2>Calibration Analysis</h2>
        <p>Expected Calibration Error: <span class="metric-value">{{ "%.4f"|format(calibration_analysis.ece) }}</span></p>
    </div>
    {% endif %}

    {% if slice_analysis %}
    <div class="section">
        <h2>Slice Analysis</h2>
        {% for slice_dim, slice_data in slice_analysis.items() %}
        <h3>{{ slice_dim }}</h3>
        <table>
            <tr><th>Slice</th><th>ECE</th><th>Count</th></tr>
            {% for slice_val, data in slice_data.items() %}
            <tr>
                <td>{{ slice_val }}</td>
                <td>{{ "%.4f"|format(data.ece) }}</td>
                <td>{{ data.bin_counts|sum }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}
    </div>
    {% endif %}

</body>
</html>
"""


class ReportGenerator:
    """Generates PDF reports from evaluation results."""

    def __init__(self, config: dict):
        """Initialize the report generator.

        Args:
            config: Reporting configuration section.
        """
        self.output_dir = Path(config.get("output_dir", "./outputs/reports"))
        self.template_dir = Path(config.get("template_dir", "src/reporting/templates"))
        self.include_sections = config.get(
            "include_sections",
            ["executive_summary", "methodology", "model_comparison", "calibration_analysis"],
        )

    def generate(
        self,
        report_data: dict,
        output_filename: str = "evaluation_report.pdf",
    ) -> Path:
        """Generate a PDF report from evaluation data.

        Args:
            report_data: Dict containing all report sections:
                - executive_summary: str
                - methodology: str
                - metrics: dict[model_name, dict[metric, value]]
                - feature_importance: list of (name, value) tuples
                - figures: dict[name, path]
                - calibration_analysis: dict
                - slice_analysis: dict

        Returns:
            Path to the generated PDF.
        """
        ensure_dir(self.output_dir)

        # Render HTML from template
        html_content = self._render_html(report_data)

        # Save HTML (always useful as a fallback)
        html_path = self.output_dir / output_filename.replace(".pdf", ".html")
        with open(html_path, "w") as f:
            f.write(html_content)
        logger.info("HTML report saved to %s", html_path)

        # Generate PDF
        pdf_path = self.output_dir / output_filename
        try:
            from weasyprint import HTML

            HTML(string=html_content, base_url=str(self.output_dir)).write_pdf(
                str(pdf_path)
            )
            logger.info("PDF report saved to %s", pdf_path)
        except ImportError:
            logger.warning(
                "weasyprint not installed. HTML report saved but PDF generation skipped."
            )
            return html_path

        return pdf_path

    def _render_html(self, report_data: dict) -> str:
        """Render the report HTML from template and data.

        Args:
            report_data: Report data dict.

        Returns:
            Rendered HTML string.
        """
        # Try to load custom template, fall back to built-in
        if self.template_dir.exists() and (self.template_dir / "report.html").exists():
            env = Environment(loader=FileSystemLoader(str(self.template_dir)))
            template = env.get_template("report.html")
        else:
            env = Environment()
            template = env.from_string(DEFAULT_TEMPLATE)

        # Prepare template context
        context = {
            "executive_summary": report_data.get("executive_summary"),
            "methodology": report_data.get("methodology"),
            "metrics": report_data.get("metrics"),
            "metric_names": [],
            "feature_importance": report_data.get("feature_importance", []),
            "figures": report_data.get("figures", {}),
            "calibration_analysis": report_data.get("calibration_analysis"),
            "slice_analysis": report_data.get("slice_analysis"),
        }

        # Extract metric names from first model
        if context["metrics"]:
            first_model = next(iter(context["metrics"].values()))
            context["metric_names"] = list(first_model.keys())

        return template.render(**context)
