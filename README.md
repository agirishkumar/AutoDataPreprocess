# AutoDataPreprocess

AutoDataPreprocess is a comprehensive Python library for automated data preprocessing. It provides a wide range of tools and techniques to clean, transform, and prepare data for machine learning models.

## Features

- Data loading from various sources (CSV, JSON, Excel, HTML, XML, Pickle, SQL, API)
- Basic data analysis and visualization
- Data cleaning (handling missing values, outliers, duplicates)
- Feature engineering
- Encoding of categorical variables (Onehot, label, ordinal, target, woe, james_stein, catboost, binary)
- Scaling and normalization
- Dimensionality reduction
- Feature selection
- Handling imbalanced data
- Time series preprocessing
- Data anonymization

## Installation

You can install AutoDataPreprocess using pip: `pip install autodatapreprocess`

## Quick Start

```python
from autodatapreprocess import AutoDataPreprocess

# Load data
adp = AutoDataPreprocess('your_data_file.csv')

# Perform basic analysis
adp.basic_analysis()

# Clean the data
cleaned_data = adp.clean(missing='mean', outliers='iqr')

# Perform feature engineering
engineered_data = adp.fe(target_column='target', polynomial_degree=2)

# Encode categorical variables
encoded_data = adp.encode(methods={'category_column': 'onehot'})

# Scale the data
scaled_data = adp.scale(method='standard')

```

## Detailed Usage

### Data Loading
Load data from various sources:

```python
# From CSV
adp = AutoDataPreprocess('data.csv')

# From SQL
adp = AutoDataPreprocess(sql_query="SELECT * FROM table", sql_connection_string="your_connection_string")

# From API
adp = AutoDataPreprocess(api_url="https://api.example.com/data", api_params={"key": "value"})
```
### Data Cleaning
Clean your data with various options:

```python
cleaned_data = adp.clean(
    missing='mean',
    outliers='iqr',
    drop_threshold=0.7,
    date_format='%Y-%m-%d',
    remove_duplicates=True
)
```

### Feature Engineering
Perform feature engineering:

```python
engineered_data = adp.fe(
    target_column='target',
    polynomial_degree=2,
    interaction_only=False,
    bin_numeric=True,
    num_bins=5,
    cyclical_features=['month', 'day_of_week'],
    text_columns=['description'],
    date_columns=['date']
)
```

### Encoding
Encode categorical variables:

```python
encoded_data = adp.encode(
    methods={
        'category1': 'onehot',
        'category2': 'label',
        'category3': 'target'
    },
    target_column='target'
)
```

### Scaling and Normalization
Scale or normalize your data:

```python
scaled_data = adp.scale(method='standard')
normalized_data = adp.normalize(method='l2')
```

### Dimensionality Reduction
Reduce the dimensionality of your data:

```python
reduced_data = adp.dimreduction(method='pca', n_components=5)
```

### Feature Selection
Select the most important features:

```python
selected_data = adp.feature_selection(
    target_column='target',
    method='correlation',
    correlation_threshold=0.8
)
```

### Handling Imbalanced Data
Balance your dataset:

```python
balanced_data = adp.balance_data(
    target_column='target',
    method='smote',
    sampling_strategy='auto'
)
```

### Time Series Preprocessing
Preprocess time series data:

```python
preprocessed_ts_data = adp.time_series_preprocessing(
    time_column='date',
    freq='D',
    method='mean',
    detrend_columns=['value'],
    seasonality_columns=['value'],
    lag_columns=['value'],
    lags=[1, 7, 30]
)
```

### Data Anonymization
Anonymize sensitive data:

```python
anonymized_data = adp.apply_anonymization(
    columns=['sensitive_column'],
    method='hash',
    hash_algorithm='sha256'
)
```