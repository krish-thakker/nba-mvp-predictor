# NBA MVP Predictor

**A Random Forest–based tool to fetch NBA stats, train an MVP prediction model, and evaluate performance.**

## Project Overview

This Python project leverages the [NBA API](https://github.com/swar/nba_api) and scikit-learn to:
- Fetch and preprocess NBA player and team statistics for regular seasons.
- Label seasons with actual MVP winners.
- Train a Random Forest classifier to predict NBA MVPs based on per-game and team performance metrics.
- Evaluate model performance on a hold-out season.
- Generate season-by-season MVP “picks” and feature importances.

## Features

- **Automatic Data Retrieval**  
  Fetches per-game player stats (including Plus/Minus) and team win totals via `nba_api` with retry logic.

- **Data Preprocessing**  
  Filters eligible players (≥65 games), merges player and team data, and maps seasons to known MVPs.

- **Model Training & Evaluation**  
  - Splits out the most recent season for testing.
  - Trains a Random Forest with balanced class weights.
  - Reports classification metrics and cross-validated macro F1 score.

- **Season MVP Picks**  
  Iteratively retrains and predicts MVP for each season to compute accuracy over historical data.

- **Feature Importance Analysis**  
  Outputs both Gini importance and permutation importance for model interpretability.

## Installation

1. **Clone the repository**  
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

Listed in `requirements.txt`:

```
certifi==2025.4.26
charset-normalizer==3.4.2
idna==3.10
joblib==1.5.1
nba_api==1.9.0
numpy==2.2.6
pandas==2.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.3
scikit-learn==1.6.1
scipy==1.15.3
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2
urllib3==2.4.0
```

## Usage

Run the prediction script:

```bash
python main.py
```

This will:
- Fetch data for seasons 2009–10 through 2023–24.
- Train and evaluate the model.
- Print classification reports, CV macro-F1, MVP picks with accuracy, and feature importances.

## Configuration

- **Seasons**  
  Modify the `seasons` list in `main.py` to include or exclude seasons.
- **Model Parameters**  
  Adjust `RandomForestClassifier` hyperparameters (e.g., `n_estimators`, `max_depth`) in the script.
- **API Settings**  
  Change `API_TIMEOUT`, `RETRY_DELAY`, and `MAX_RETRIES` for data fetching behavior.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for:
- Adding more advanced models.
- Improving feature engineering.
- Supporting playoff data and awards beyond MVP.

## Acknowledgements

- Data provided by the NBA API.
- Model inspired by exploratory NBA analytics projects.
