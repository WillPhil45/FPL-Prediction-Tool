# FPL Prediction Tool

A sophisticated machine learning system for Fantasy Premier League (FPL) player performance prediction and team optimization. This project combines multiple ML algorithms, advanced feature engineering, and genetic algorithms to create optimal FPL teams.

## Project Overview

This system predicts FPL player performance and generates optimal team selections using:
- **6 Different ML Algorithms**: Random Forest, Gradient Boosting, XGBoost, Ridge Regression, Deep Learning, and LSTM networks
- **Position-Specific Modeling**: Separate models for Goalkeepers, Defenders, Midfielders, and Forwards
- **Advanced Feature Engineering**: 75+ features including time-series rolling windows and form indicators
- **Genetic Algorithm Optimization**: Constraint-aware team selection respecting FPL rules
- **Comprehensive Evaluation**: Validated across 30+ gameweeks with performance tracking

## Key Features

### Machine Learning Pipeline
- **Multi-Algorithm Approach**: Ensemble of 6 different ML algorithms with hyperparameter optimization
- **Position-Aware Modeling**: Tailored models for each player position to capture position-specific patterns
- **Time-Series Features**: Rolling averages, form indicators, and temporal patterns
- **Neural Networks**: Deep learning models with GPU acceleration and LSTM networks with attention mechanisms

### Advanced Analytics
- **Model Interpretability**: SHAP and LIME explanations for prediction insights
- **Performance Ranking**: Comprehensive model comparison using multiple metrics
- **Feature Importance**: Position-specific feature attribution analysis
- **Validation**: Time-series cross-validation maintaining temporal order

### Team Optimization
- **Genetic Algorithm**: Population-based search with constraint handling for FPL rules
- **Multi-Objective**: Maximizes predicted points while satisfying team composition constraints
- **Advanced Operators**: Position-aware crossover and mutation with repair mechanisms

## Performance Results

Current algorithm performance ranking based on comprehensive evaluation:
1. **Random Forest Regressor** - Best overall performance
2. **Deep Learning** - Strong performance with neural networks
3. **LSTM** - Excellent for capturing temporal patterns
4. **Ridge Regression** - Reliable baseline performance
5. **Gradient Boosting** - Good ensemble performance
6. **XGBoost** - Advanced gradient boosting

## Quick Start

### Prerequisites
```bash
# Core ML libraries
pip install pandas numpy scikit-learn tensorflow xgboost
pip install optuna shap lime matplotlib seaborn

# Additional dependencies
pip install requests beautifulsoup4
```

### Basic Usage

1. **Train Models**:
```bash
python scripts/model_training.py
```

2. **Generate Predictions**:
```bash
# Single gameweek
python scripts/prediction.py --gw 1

# Multiple gameweeks
python scripts/prediction.py --gw-range "1-10"

# All available gameweeks
python scripts/prediction.py --all-gws
```

3. **Optimize Team Selection**:
```bash
python scripts/fpl_team_optimizer.py
```

### Advanced Options

```bash
# Parallel processing
python scripts/prediction.py --gw-range "1-10" --parallel

# Skip explainability analysis for faster processing
python scripts/prediction.py --gw 1 --skip-explain

# Custom output directory
python scripts/prediction.py --gw 1 --output-dir custom_results/
```

## Project Structure

```
FPL Prediction Tool/
├── scripts/                    # Core ML and analysis scripts
│   ├── model_training.py       # Train all ML models
│   ├── prediction.py          # Generate predictions
│   ├── fpl_team_optimizer.py  # Team optimization
│   └── ...
├── data/                      # Training and prediction data
│   ├── merged_gw_clean.csv    # Main dataset
│   ├── time_*.csv            # Time-based splits
│   └── player_predictions_data_gw*.csv
├── models/                    # Trained models and metadata
│   ├── regression_models.pkl
│   ├── checkpoints/          # Neural network checkpoints
│   └── metadata/
├── predictions/               # Prediction results by gameweek
│   ├── gw01/, gw02/, ...     # Gameweek-specific results
│   └── summary/              # Performance summaries
├── analysis_results/          # Performance analysis and plots
├── experimental_results/      # Training plots and metrics
└── Up-to-data/               # FPL data repository
```

## Configuration

### Model Training Parameters
- **Hyperparameter Optimization**: Optuna with 20-30 trials per model
- **Neural Networks**: GPU acceleration with mixed precision training
- **LSTM Networks**: Bidirectional architecture with multiple lookback windows
- **Early Stopping**: Prevents overfitting with patience mechanisms

### Team Optimization Settings
- **Population Size**: 100-150 individuals per generation
- **Constraint Handling**: Enforces FPL rules (11 players, position limits)
- **Genetic Operators**: Position-aware crossover and mutation

## Features and Data

### Core FPL Features (75+ features)
- **Performance Metrics**: Goals, assists, bonus points, clean sheets, saves
- **Contextual Data**: Minutes played, opponent team, home/away status
- **Economic Metrics**: Player value, transfers in/out, selection percentage
- **Form Indicators**: Rolling averages and momentum features

### Advanced Feature Engineering
- **Time-Series Windows**: Last game, 3-game, 10-game rolling averages
- **Cumulative Metrics**: Season totals and per-minute statistics
- **Position Encoding**: One-hot encoded position indicators
- **Fixture Analysis**: Difficulty ratings and availability

## Model Architecture

### Regression Models (Primary)
- **Random Forest**: Tree-based ensemble with feature importance
- **Gradient Boosting**: Sequential boosting with MSE optimization
- **XGBoost**: Advanced gradient boosting with extensive hyperparameter tuning
- **Ridge**: Linear regression with L2 regularization
- **Deep Learning**: 2-4 layer networks with batch normalization and dropout
- **LSTM**: Bidirectional LSTM with attention mechanisms

### Classification Models (Supporting)
- **Player Availability**: Predicts `played_next_match` and `started_next_match`

## Evaluation Metrics

### Regression Performance
- **MSE, MAE, RMSE**: Standard regression metrics
- **R²**: Coefficient of determination
- **Position-Specific**: Separate evaluation for each position

### Team Quality Assessment
- **Points Differential**: Comparison vs actual optimal team
- **Player Accuracy**: Percentage of correctly predicted players
- **Formation Accuracy**: Tactical setup prediction

### Model Comparison
- **Ranking Metrics**: Kendall's tau, NDCG
- **Statistical Testing**: Multi-gameweek validation
- **Consistency Analysis**: Performance stability across seasons

## 🔍 Model Interpretability

### SHAP Analysis
- **Global Explanations**: Feature importance across all predictions
- **Local Explanations**: Individual prediction breakdowns
- **Position-Specific**: Separate analysis for each playing position

### LIME Integration
- **Local Explanations**: Alternative interpretation method
- **Feature Perturbation**: Understanding prediction sensitivity

## Analysis Scripts

```bash
# Feature importance analysis
python scripts/improved_feature_attribution.py

# Model ranking and comparison
python scripts/ranking_tests.py

# Average score calculations
python scripts/average_FPL_score.py

# Neural network stability testing
python scripts/NN_stochasticity_test.py
```

## License

This project is for educational and research purposes. FPL data is sourced from the Fantasy-Premier-League repository.

## Acknowledgments

- **Fantasy-Premier-League Repository**: Primary data source for FPL statistics
- **Optuna**: Hyperparameter optimization framework
- **SHAP/LIME**: Model interpretability libraries
- **TensorFlow/PyTorch**: Deep learning frameworks

*This project demonstrates the application of advanced machine learning techniques to fantasy sports analytics, combining traditional ML, deep learning, and optimization algorithms in a production-ready system.*