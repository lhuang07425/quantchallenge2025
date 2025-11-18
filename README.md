# QuantChallenge 2025 - Research Competition Submission

## Competition Overview

QuantChallenge 2025 is an elite quantitative finance research and trading competition that challenges participants to develop predictive models and trading strategies using real-world market data. The competition consists of two main components:

- **Research Track**: Building machine learning models to predict market behavior from historical data
- **Trading Track**: Implementing algorithmic trading strategies in a live simulation environment

This repository contains my submission for the **Research Track**, where I developed multi-output regression models to predict two target variables (Y1 and Y2) from a complex feature set.

## Achievement

🏆 **Top 9% Finish** - Placed in the top 9% among all participants in the research competition

## Project Structure

```
.
├── scikit_model.ipynb          # Main model development notebook
├── scikit_prediction.csv       # Final predictions submission
├── train.csv                   # Training dataset (not included)
├── test.csv                    # Test dataset (not included)
├── train_new.csv              # Additional training features (not included)
└── test_new.csv               # Additional test features (not included)
```

## Approach

### Data Preprocessing

The competition provided market research data with the following characteristics:
- **Features**: 17 columns including time, and features A through P
- **Targets**: Two continuous variables (Y1, Y2) requiring multi-output regression
- **Challenge**: Data split across multiple files requiring careful merging

**Important Note**: The CSV datasets were intentionally designed to test participants' data handling skills and were not meant to be used directly. The proper approach required:
1. Joining supplementary features (O and P) from separate files
2. Converting time data to numeric format (Unix timestamp)
3. Handling missing values appropriately

### Model Architecture

I implemented a robust scikit-learn pipeline with the following components:

```python
Pipeline([
    ('preprocess', ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), feature_cols)
    ])),
    ('regressor', MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=42)
    ))
])
```

**Key Technical Decisions**:
- **HistGradientBoostingRegressor**: Fast, memory-efficient gradient boosting that handles missing values natively
- **MultiOutputRegressor**: Enables simultaneous prediction of both Y1 and Y2 targets
- **Median Imputation**: Robust handling of missing values in the feature space
- **Time Encoding**: Converted datetime to Unix timestamps for numerical processing

### Why This Approach Works

1. **Gradient Boosting Excellence**: Histogram-based gradient boosting is particularly effective for:
   - Capturing non-linear relationships in financial data
   - Handling mixed feature types without extensive preprocessing
   - Fast training on large datasets (80,000 training samples)

2. **Multi-Output Learning**: Training a single model for both targets allows the algorithm to:
   - Learn shared patterns between Y1 and Y2
   - Reduce overfitting through regularization across outputs
   - Maintain consistent feature importance

3. **Minimal Feature Engineering**: By letting the gradient boosting algorithm learn feature interactions automatically, we avoid:
   - Overfitting to training data through manual feature creation
   - Information leakage from looking at test data
   - Complexity that reduces model interpretability

## Results

The model achieved a top 9% ranking on the private leaderboard, demonstrating:
- Strong generalization to unseen data
- Robust handling of the multi-output prediction task
- Effective feature utilization without overfitting

## Key Insights

1. **Data Integration Matters**: Properly joining the supplementary features (O, P) from separate files was crucial for competitive performance
2. **Simple is Better**: A straightforward pipeline with strong base algorithms outperformed complex ensemble approaches
3. **Time Encoding**: Converting time to numeric format allowed the model to capture temporal patterns effectively
4. **Missing Value Strategy**: Median imputation proved more robust than mean imputation for this financial dataset

## Technical Stack

- **Python 3.12**
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations
- **scikit-learn 1.5**: Machine learning pipeline
  - `HistGradientBoostingRegressor`: Core model
  - `MultiOutputRegressor`: Multi-target wrapper
  - `SimpleImputer`: Missing value handling
  - `ColumnTransformer`: Feature preprocessing
  - `Pipeline`: End-to-end workflow

## Reproducibility

To reproduce these results:

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

2. Ensure all data files are in the correct location:
   - `train.csv` and `test.csv` (main datasets)
   - `train_new.csv` and `test_new.csv` (supplementary features)

3. Run the notebook:
```bash
jupyter notebook scikit_model.ipynb
```

4. The model will:
   - Load and merge all data files
   - Preprocess features (time conversion, imputation)
   - Train the multi-output regressor
   - Generate `scikit_prediction.csv` with predictions


**Note**: Competition data is proprietary and not included in this repository. This README serves as documentation of the methodology and approach used.