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

def test_basic_analysis_correlation(sample_csv_file,capsys, monkeypatch):
    # Mock the plt.show() function to avoid displaying plots
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    adp = AutoDataPreprocess(sample_csv_file)
    adp.basic_analysis()
    captured = capsys.readouterr()
    print("Captured output:")
    print(captured.out)
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