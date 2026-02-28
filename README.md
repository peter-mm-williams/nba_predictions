## NBA Prop Prediction Framework - Complete Information
Show less
### 1. Project Metadata (pyproject.toml)
- **Name**: nba-prop-predictor
- **Version**: 0.1.0
- **Description**: NBA player prop prediction framework using sequence and tabular models
- **License**: MIT
- **Python**: >=3.10
### 2. Core Dependencies (requirements.txt)
**Core Libraries:**
- nba_api>=1.4
- pandas>=2.0
- numpy>=1.24
- torch>=2.0
- xgboost>=2.0
- scikit-learn>=1.3
- optuna>=3.4
- pyarrow>=14.0
**Configuration & Data:**
- pyyaml>=6.0
- requests>=2.31
- beautifulsoup4>=4.12
**Visualization & Reporting:**
- matplotlib>=3.8
- seaborn>=0.13
- jinja2>=3.1
- weasyprint>=60.0
**Evaluation & Utilities:**
- properscoring>=0.1
- tqdm>=4.66
**Development:**
- pytest>=7.4, pytest-cov>=4.1, pytest-mock>=3.12
- black>=23.0, ruff>=0.1, mypy>=1.7
- ipykernel>=6.27, jupyter>=1.0
### 3. Configuration System (config/config.yaml)
The framework uses a comprehensive YAML-based configuration with environment variable override support using `${VAR_NAME:default}` syntax.
**Key Configuration Sections:**
**Paths:**
```yaml
paths:
  data_dir: ./data
  output_dir: ./outputs
  model_dir: ./outputs/models
  log_dir: ./logs
```
**Data Acquisition:**
```yaml
data:
  seasons:
    start: 2015
    end: 2024 (configurable via CURRENT_SEASON env var)
  sources:
    nba_api:
      rate_limit_seconds: 0.6
      retry_attempts: 3
      timeout_seconds: 30
    basketball_reference:
      enabled: true
      rate_limit_seconds: 3.0
```
**Data Cleaning:**
```yaml
cleaning:
  min_minutes_played: 5
  min_games_for_player: 20
  handle_trades: split_season
  exclude_playoff_games: false
```
**Feature Engineering:**
```yaml
features:
  rolling_windows: [3, 5, 10, 20]
  target_stats:
    - points
    - rebounds
    - three_pointers_made
    - free_throws_made
  temporal_features:
    - days_rest
    - back_to_back
    - home_away
    - season_game_number
  opponent_features:
    - defensive_rating_season
    - pace_season
    - position_defense_rank
  player_features:
    - season_average
    - career_average
    - usage_rate
    - minutes_trend
  team_features:
    - team_pace
    - team_offensive_rating
    - teammate_injury_impact
  sequence_length: 10
```
**Model Configurations:**
```yaml
models:
  distribution:
    type: negative_binomial
    max_value: 60
    num_bins: 61
  
  tabular:
    logistic:
      solver: lbfgs
      max_iter: 1000
      class_weight: balanced
    xgboost:
      n_estimators: 500 (env: XGB_ESTIMATORS)
      max_depth: 6
      learning_rate: 0.05
      early_stopping_rounds: 50
      eval_metric: logloss
  
  sequential:
    embedding_dim: 64
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2
    lstm:
      bidirectional: false
    transformer:
      num_heads: 4
      feedforward_dim: 256
```
**Training Settings:**
```yaml
training:
  batch_size: 64 (env: BATCH_SIZE)
  epochs: 100 (env: EPOCHS)
  learning_rate: 0.001 (env: LR)
  weight_decay: 0.0001
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  
  splits:
    train_end_season: 2022
    val_end_season: 2023
    test_end_season: 2024
  
  device: cuda (env: DEVICE)
  num_workers: 4 (env: NUM_WORKERS)
  pin_memory: true
  
  hyperopt:
    enabled: false
    n_trials: 50
    timeout_hours: 4
```
**Evaluation Metrics:**
- nll (Negative Log Likelihood)
- crps (Continuous Ranked Probability Score)
- calibration_error
- mae (Mean Absolute Error)
- rmse (Root Mean Squared Error)
**Calibration:**
- Stratification by: position, usage_tier, home_away, rest_days, opponent_strength
**Reporting:**
- Format: PDF
- Sections: executive_summary, methodology, feature_importance, model_comparison, calibration_analysis, slice_analysis, appendix
### 4. Available Make Commands (Makefile)
```
make install      - Install dependencies and editable package
make test         - Run full test suite with coverage
make test-fast    - Run tests excluding slow and integration tests
make download     - Download NBA data
make train        - Train models
make evaluate     - Evaluate trained models
make report       - Generate PDF report
make all          - Run download → train → evaluate → report pipeline
make clean        - Remove cache and compiled files
make lint         - Run ruff and mypy checks
make format       - Format code with black and ruff
```
### 5. Configuration System Implementation (src/config/settings.py)
The `Config` class provides:
- **YAML Loading**: Loads configuration from YAML files
- **Environment Variable Resolution**: Supports `${VAR_NAME:default}` syntax with automatic type casting (int/float)
- **Override Files**: Can merge override YAML files (e.g., config.dev.yaml)
- **Deep Merging**: Merges nested configuration dictionaries
- **Dot-Notation Access**: `config.get("training.batch_size")` or dictionary-style access
- **Flattening**: Can convert nested config to flat dot-separated keys
### 6. Directory Structure
```
src/
├── config/
│   ├── __init__.py
│   └── settings.py              # Config loader with env var support
├── data/
│   ├── __init__.py
│   ├── clean.py                 # Data cleaning utilities
│   ├── download.py              # NBA API data downloader
│   ├── features.py              # Feature engineering
│   ├── scraper.py               # Basketball Reference scraper
│   └── sequences.py             # Sequence building for RNNs
├── evaluation/
│   ├── __init__.py
│   ├── calibration.py           # Calibration analysis
│   ├── metrics.py               # Evaluation metrics
│   └── plots.py                 # Visualization
├── models/
│   ├── __init__.py
│   ├── base.py                  # Abstract base class for all models
│   ├── sequential/
│   │   ├── __init__.py
│   │   ├── distributions.py     # Distribution output heads
│   │   ├── lstm.py              # LSTM model
│   │   └── transformer.py       # Transformer model
│   └── tabular/
│       ├── __init__.py
│       ├── logistic.py          # Logistic regression
│       └── xgboost_model.py     # XGBoost implementation
├── reporting/
│   ├── __init__.py
│   ├── pdf_report.py            # PDF report generation
│   └── templates/
├── training/
│   ├── __init__.py
│   ├── callbacks.py             # Training callbacks
│   ├── hyperopt.py              # Hyperparameter optimization
│   └── trainer.py               # Neural network trainer
└── utils/
    ├── __init__.py
    ├── io.py                    # File I/O utilities
    ├── logging.py               # Logging setup
    └── temporal.py              # Time-based utilities
tests/
├── __init__.py
├── conftest.py
├── data/
│   ├── test_clean.py
│   ├── test_download.py
│   ├── test_features.py
│   └── test_sequences.py
├── evaluation/
│   └── test_metrics.py
├── fixtures/
│   └── sample_box_scores.json
├── integration/
│   └── test_pipeline.py
└── models/
    ├── test_sequential.py
    └── test_tabular.py
scripts/
├── run_pipeline.py              # Main pipeline execution
├── download_data.py             # Data download CLI
└── generate_report.py           # Report generation
config/
├── config.yaml                  # Main configuration
├── config.dev.yaml              # Development overrides
└── config.prod.yaml             # Production overrides
notebooks/
├── 01_data_exploration.ipynb
├── 02_feature_analysis.ipynb
└── 03_model_comparison.ipynb
```
### 7. CLI Usage - run_pipeline.py
**Purpose**: Main pipeline execution script for orchestrating the entire prediction workflow.
**Basic Usage:**
```bash
python scripts/run_pipeline.py --config config/config.yaml
```
**Advanced Usage:**
```bash
# Run specific stages
python scripts/run_pipeline.py --config config/config.yaml --stages download,clean,features
# Train specific model on specific stat
python scripts/run_pipeline.py --config config/config.yaml --stages train --model lstm --stat points
# With override config
python scripts/run_pipeline.py --config config/config.yaml --override config/config.dev.yaml
```
**Environment Variables:**
```
ENV=production
DEVICE=cuda
BATCH_SIZE=128
```
**Available Stages:**
```
["download", "clean", "features", "sequences", "train", "evaluate"]
```
**Model Choices:**
- xgboost (default)
- logistic
- lstm
- transformer
**Target Stats:**
- points (default)
- rebounds
- three_pointers_made
- free_throws_made
**Stage Details:**
1. **download**: Fetches data from NBA API and Basketball Reference
2. **clean**: Applies data cleaning rules (min minutes, trades handling, etc.)
3. **features**: Builds temporal, opponent, player, and team features
4. **sequences**: Formats data into sequences for LSTM/Transformer models
5. **train**: Trains model on appropriate data split (2015-2022 train, 2022-2023 val, 2023-2024 test)
6. **evaluate**: Computes metrics (NLL, CRPS, calibration error, MAE, RMSE) on test set
### 8. Data Download CLI - download_data.py
**Purpose**: Standalone data download utility.
**Basic Usage:**
```bash
python scripts/download_data.py --config config/config.yaml
```
**Custom Season Range:**
```bash
python scripts/download_data.py --config config/config.yaml --seasons 2022 2024
```
**Skip Basketball Reference:**
```bash
python scripts/download_data.py --config config/config.yaml --skip-bbref
```
**Options:**
- `--config`: Path to configuration file (default: config/config.yaml)
- `--override`: Path to override configuration file
- `--seasons`: Start and end season years (e.g., 2022 2024)
- `--skip-bbref`: Skip Basketball Reference scraping
**Data Sources:**
- **NBA API**: Box scores, play-by-play, player stats
- **Basketball Reference**: Supplementary advanced stats
### 9. Model Interface - src/models/base.py
**BaseStatPredictor Abstract Class**
All models inherit from `BaseStatPredictor` and must implement:
```python
class BaseStatPredictor(ABC):
    # Required abstract methods:
    
    def fit(self, train_data, val_data, config) -> BaseStatPredictor:
        """Train the model. Returns self for method chaining."""
    
    def predict_distribution(self, X):
        """Return full probability distribution over stat values."""
    
    def predict_proba_over(self, X, threshold: int) -> np.ndarray:
        """Return P(stat > threshold) for each sample."""
    
    def save(self, path: str) -> None:
        """Save model to disk."""
    
    @classmethod
    def load(cls, path: str) -> BaseStatPredictor:
        """Load model from disk."""
    
    # Optional convenience methods:
    
    def predict_mean(self, X) -> np.ndarray:
        """Return mean (expected value) of predicted distribution."""
    
    def predict_pmf(self, X, max_value: int = 60) -> np.ndarray:
        """Return probability mass function over [0, max_value].
        Shape: (n_samples, max_value + 1)"""
```
**Key Design Principles:**
- All models output **probability distributions**, not point predictions
- Supports comparison between tabular (XGBoost, Logistic) and sequential (LSTM, Transformer) models
- Consistent PMF interface for evaluation and calibration
### 10. Distribution Output - src/models/sequential/distributions.py
**NegativeBinomialDistribution Class**
Parameterized by mean (mu) and dispersion (alpha):
```
PMF: P(X=k) = Gamma(k+r) / (Gamma(k+1)*Gamma(r)) * p^r * (1-p)^k
where r = 1/alpha, p = 1/(1 + mu*alpha)
```
**Methods:**
- `log_prob(x)`: Compute log probability of observed values
- `cdf(x)`: Compute cumulative distribution function
- `mean()`: Return distribution mean
- `variance()`: Return distribution variance (mu + alpha * mu²)
- `pmf_range(max_value)`: Compute PMF for values 0 through max_value
  - Returns shape: (batch_size, max_value + 1)
**DistributionHead Class**
Neural network output head for Negative Binomial:
- Learns mean (mu) via softplus activation
- Learns dispersion (alpha) via softplus + small epsilon
- `forward(x)`: Returns NegativeBinomialDistribution instance
- `predict_proba(x, threshold)`: Computes P(stat > threshold)
**DiscretizedHead Class**
Alternative output head using categorical distribution over bins:
- Projects to num_bins (default 61 for 0-60 range)
- Uses log-softmax for log-probabilities
- `forward(x)`: Returns log-probabilities
- `predict_proba(x, threshold)`: Sums probabilities for bins > threshold
---
This comprehensive information covers:
1. Complete project metadata and dependencies
2. Full configuration schema with all available options
3. CLI usage for both pipeline execution and data download
4. Directory structure and code organization
5. Model interface and design principles
6. Distribution output specifications
7. All available Make commands
You now have all the details needed to write a thorough, professional README that covers setup, configuration, usage, architecture, and development.
