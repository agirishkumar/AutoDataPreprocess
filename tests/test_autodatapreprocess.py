import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
import matplotlib.pyplot as plt
from AutoDataPreprocess import AutoDataPreprocess

@pytest.fixture
def sample_csv_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("A,B,C\n1,2,3\n4,5,6\n7,8,9\n10,,12\n13,14,15")
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_json_file():
    data = {
        "A": [1, 4, 7, 10, 13],
        "B": [2, 5, 8, None, 14],
        "C": [3, 6, 9, 12, 15]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_excel_file():
    df = pd.DataFrame({
        "A": [1, 4, 7, 10, 13],
        "B": [2, 5, 8, None, 14],
        "C": [3, 6, 9, 12, 15]
    })
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        df.to_excel(f.name, index=False)
    yield f.name
    os.unlink(f.name)

def test_load_csv(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    assert isinstance(adp.data, pd.DataFrame)
    assert adp.data.shape == (5, 3)
    assert list(adp.data.columns) == ['A', 'B', 'C']
    assert adp.data['A'].dtype == 'int64'
    assert adp.data['B'].dtype == 'float64'  # Due to missing value
    assert adp.data['C'].dtype == 'int64'
    assert pd.isna(adp.data.loc[3, 'B'])

def test_load_json(sample_json_file):
    adp = AutoDataPreprocess(sample_json_file)
    assert isinstance(adp.data, pd.DataFrame)
    assert adp.data.shape == (5, 3)
    assert list(adp.data.columns) == ['A', 'B', 'C']
    assert adp.data['A'].dtype == 'int64'
    assert adp.data['B'].dtype == 'float64'  # Due to None value
    assert adp.data['C'].dtype == 'int64'
    assert pd.isna(adp.data.loc[3, 'B'])

def test_load_excel(sample_excel_file):
    adp = AutoDataPreprocess(sample_excel_file)
    assert isinstance(adp.data, pd.DataFrame)
    assert adp.data.shape == (5, 3)
    assert list(adp.data.columns) == ['A', 'B', 'C']
    assert adp.data['A'].dtype == 'int64'
    assert adp.data['B'].dtype == 'float64'  # Due to None value
    assert adp.data['C'].dtype == 'int64'
    assert pd.isna(adp.data.loc[3, 'B'])

def test_load_unsupported():
    with tempfile.NamedTemporaryFile(suffix='.unsupported') as f:
        with pytest.raises(ValueError, match="Unsupported file type"):
            AutoDataPreprocess(f.name)

def test_load_nonexistent_file():
    with pytest.raises(ValueError, match="File does not exist"):
        AutoDataPreprocess('nonexistent_file.csv')

def test_basic_analysis(sample_csv_file, capsys):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.basic_analysis()
    captured = capsys.readouterr()
    
    # Check if key phrases are in the output
    assert "Dataset Information:" in captured.out
    assert "Shape of the dataset: (5, 3)" in captured.out
    assert "Summary Statistics:" in captured.out
    assert "Data Types:" in captured.out
    assert "Missing Value Analysis:" in captured.out
    assert "Number of duplicate rows: 0" in captured.out
    assert "Skewness Analysis:" in captured.out
    assert "Unique Values Count:" in captured.out
    
    # Check specific statistics
    assert "B               1        20.0" in captured.out  # Missing value counts
    assert "A       5 non-null" in captured.out
    assert "C       5 non-null" in captured.out
    
    # Check data types
    assert "float64    1" in captured.out
    assert "int64      2" in captured.out
    
    # Check summary statistics
    assert "mean" in captured.out
    assert "std" in captured.out
    assert "min" in captured.out
    assert "25%" in captured.out
    assert "50%" in captured.out
    assert "75%" in captured.out
    assert "max" in captured.out
    
    # Close all plots to clean up
    plt.close('all')

def test_basic_analysis_correlation(sample_csv_file, monkeypatch):
    # Mock the plt.show() function to avoid displaying plots
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    adp = AutoDataPreprocess(sample_csv_file)
    adp.basic_analysis()
    
    # Check if correlation heatmap was created
    figures = [plt.figure(num) for num in plt.get_fignums()]
    assert any('Correlation Heatmap' in ax.get_title() for fig in figures for ax in fig.get_axes()), "Correlation Heatmap not found"

    # Close all plots to clean up
    plt.close('all')

def test_basic_analysis_distribution_plots(sample_csv_file, monkeypatch):
    # Mock the plt.show() function to avoid displaying plots
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    adp = AutoDataPreprocess(sample_csv_file)
    adp.basic_analysis()
    
    # Check if distribution plots were created
    assert any('Distribution of A' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    assert any('Distribution of B' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    assert any('Distribution of C' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    
    # Check if boxplots were created
    assert any('Boxplot of A' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    assert any('Boxplot of B' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    assert any('Boxplot of C' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    
    # Close all plots to clean up
    plt.close('all')

def test_handle_missing_values_mean(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.handle_missing_values(adp.data, strategy='mean')
    assert not adp.data.isnull().any().any(), "There should be no missing values after mean imputation"
    assert adp.data.loc[3, 'B'] == pytest.approx(7.25), "The mean of column B should be used to fill missing values"

def test_handle_missing_values_median(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.handle_missing_values(adp.data, strategy='median')
    assert not adp.data.isnull().any().any(), "There should be no missing values after median imputation"
    assert adp.data.loc[3, 'B'] == 6.5, "The median of column B should be used to fill missing values"

def test_handle_missing_values_mode(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.handle_missing_values(adp.data, strategy='mode')
    assert not adp.data.isnull().any().any(), "There should be no missing values after mode imputation"
    assert adp.data.loc[3, 'B'] == 2.0, "The mode of column B should be used to fill missing values"

def test_handle_missing_values_ffill(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.handle_missing_values(adp.data, strategy='ffill')
    assert not adp.data.isnull().any().any(), "There should be no missing values after forward fill"
    assert adp.data.loc[3, 'B'] == 8.0, "The missing value in column B should be forward filled with 8"

def test_handle_missing_values_bfill(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.handle_missing_values(adp.data, strategy='bfill')
    assert not adp.data.isnull().any().any(), "There should be no missing values after backward fill"
    assert adp.data.loc[3, 'B'] == 14.0, "The missing value in column B should be backward filled with 14"

def test_drop_columns_with_missing_values(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    df_cleaned = adp.drop_columns_with_missing_values(adp.data, drop_threshold=0.2)
    assert 'B' not in df_cleaned.columns, "Column B should be dropped due to high percentage of missing values"

def test_outlier_detection(sample_csv_file, monkeypatch):
    adp = AutoDataPreprocess(sample_csv_file)
    monkeypatch.setattr(plt, 'show', lambda: None)
    adp.outlier_analysis()
    
    # We expect boxplots to be generated for each numerical column
    assert any('Outliers in A' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    assert any('Outliers in B' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    assert any('Outliers in C' in ax.get_title() for fig in plt.get_fignums() for ax in plt.figure(fig).get_axes())
    
    plt.close('all')

def test_scaling(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.scale(method='standard')
    
    # Check if columns are scaled (mean ~ 0, std ~ 1)
    assert np.allclose(adp.data.mean(), 0, atol=1e-1), "Mean of scaled columns should be close to 0"
    assert np.allclose(adp.data.std(), 1, atol=1e-1), "Standard deviation of scaled columns should be close to 1"

def test_encoding(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.data['D'] = ['cat', 'dog', 'cat', 'bird', 'dog']
    adp.encode(methods={'D': 'onehot'})
    
    assert 'D_cat' in adp.data.columns and 'D_dog' in adp.data.columns and 'D_bird' in adp.data.columns, "One-hot encoding failed"
    assert len(adp.data.columns) == 6, "There should be 6 columns after one-hot encoding"

def test_pca_reduction(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    df_pca = adp.pca_reduction(n_components=2)
    
    assert df_pca.shape[1] == 5, "There should be 5 columns after PCA (3 original, 2 PCA components)"
    assert 'PC1' in df_pca.columns and 'PC2' in df_pca.columns, "PCA components not found in DataFrame"

def test_cleaning_pipeline(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    df_cleaned = adp.clean(missing='mean', outliers='zscore', drop_threshold=0.2, remove_duplicates=True)
    
    assert df_cleaned.shape[1] == 2, "Only columns A and C should remain after cleaning (B dropped due to missing values)"
    assert df_cleaned.shape[0] == 5, "All rows should remain after cleaning (no duplicates or extreme outliers)"

def test_time_series_preprocessing(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.data['Date'] = pd.date_range(start='2020-01-01', periods=5)
    
    df_preprocessed = adp.time_series_preprocessing(time_column='Date', freq='D', detrend_columns=['A'], 
                                                    seasonality_columns=['B'], lag_columns=['C'], lags=[1,2])
    
    assert 'A_detrended' in df_preprocessed.columns, "Detrended column A not found"
    assert 'C_lag_1' in df_preprocessed.columns and 'C_lag_2' in df_preprocessed.columns, "Lag columns for C not found"
    assert 'B_seasonal_adjusted' in df_preprocessed.columns, "Seasonal adjusted column B not found"

def test_anonymization(sample_csv_file):
    adp = AutoDataPreprocess(sample_csv_file)
    adp.data['Sensitive'] = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    
    df_anonymized = adp.apply_anonymization(columns=['Sensitive'], method='hash', hash_algorithm='sha256')
    
    assert df_anonymized['Sensitive'].nunique() == 5, "Each name should be uniquely hashed"
    assert not any(df_anonymized['Sensitive'].isin(['Alice', 'Bob', 'Charlie', 'David', 'Eve'])), "Original names should not be in the anonymized data"
