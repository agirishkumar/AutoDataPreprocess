import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mstats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from category_encoders import TargetEncoder, WOEEncoder, JamesSteinEncoder, CatBoostEncoder, BinaryEncoder
import os
import umap
from ydata_profiling import ProfileReport
from sqlalchemy import create_engine
import requests
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import detrend
from scipy.fftpack import fft
import hashlib
import random
import string


class AutoDataPreprocess:
    def __init__(self, filepath=None, sql_query=None, sql_connection_string=None, api_url=None, api_params=None):
        """
        Initializes the AutoDataPreprocess class by loading data from a file, SQL database, or API.

        Parameters:
        filepath (str): The path to the file to be loaded.
        sql_query (str): SQL query to execute if loading from a database.
        sql_connection_string (str): SQLAlchemy connection string to connect to the database.
        api_url (str): URL of the API to fetch data from.
        api_params (dict): Dictionary of query parameters to pass to the API.

        Returns:
        None
        """
        if filepath:
            self.data = self.load(filepath)
        elif sql_query and sql_connection_string:
            self.data = self.load_from_sql(sql_query, sql_connection_string)
        elif api_url:
            self.data = self.load_from_api(api_url, api_params)
        else:
            raise ValueError("Either filepath, SQL query and connection string, or API URL must be provided.")

    def load(self, filepath):
        """
        Load data from various file formats.
        
        Supported formats:
        - CSV (.csv)
        - JSON (.json)
        - Excel (.xls, .xlsx)
        - HTML (.html)
        - XML (.xml)
        - Pickle (.pkl)
        
        Args:
        filepath (str): Path to the file to be loaded.
        
        Returns:
        pandas.DataFrame: Loaded data
        
        Raises:
        ValueError: If the file type is unsupported or the file doesn't exist.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist: {filepath}")
        
        file_extension = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(filepath)
            elif file_extension == '.json':
                return pd.read_json(filepath)
            elif file_extension in ['.xls', '.xlsx']:
                return pd.read_excel(filepath)
            elif file_extension == '.html':
                return pd.read_html(filepath)[0]  
            elif file_extension == '.xml':
                return pd.read_xml(filepath)
            elif file_extension == '.pkl':
                return pd.read_pickle(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {str(e)}")
        
    def load_from_sql(self, query, connection_string):
        """
        Load data from a SQL database.

        Parameters:
        query (str): SQL query to execute.
        connection_string (str): SQLAlchemy connection string to connect to the database.

        Returns:
        pandas.DataFrame: Loaded data from SQL database.
        """
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            print("Data loaded successfully from SQL database.")
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from SQL database: {str(e)}")

    def load_from_api(self, url, params=None):
        """
        Load data from an API.

        Parameters:
        url (str): URL of the API.
        params (dict): Dictionary of query parameters to pass to the API.

        Returns:
        pandas.DataFrame: Loaded data from API.
        """
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()
            df = pd.DataFrame(data)
            print("Data loaded successfully from API.")
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from API: {str(e)}")

    def memory_usage_analysis(self):
        """
        Analyze and display memory usage of the dataset.
        """
        print("\nMemory Usage Analysis:")
        memory_usage = self.data.memory_usage(deep=True)
        print(memory_usage)
        print("\nTotal Memory Usage: {:.2f} MB".format(memory_usage.sum() / 1024 ** 2))
    
    def basic_dataset_info(self):
        """
        Print basic information about the dataset.
        """
        print("Dataset Information:")
        print(self.data.info())
        print("\nShape of the dataset:", self.data.shape)
        
    def summary_statistics(self):
        """
        Print summary statistics of the dataset.
        """
        print("\nSummary Statistics:")
        summary_stats = self.data.describe(include='all').T
        
        # Compute median only for numeric columns
        numeric_medians = self.data.select_dtypes(include=[np.number]).median()
        
        # Compute mode for all columns
        modes = self.data.mode().iloc[0]
        
        # Add median and mode to the summary statistics DataFrame
        summary_stats['median'] = numeric_medians
        summary_stats['mode'] = modes
        
        print(summary_stats)


    def data_type_analysis(self):
        """
        Analyze and print the data types of the dataset columns.
        """
        print("\nData Types:")
        print(self.data.dtypes.value_counts())

    def missing_value_analysis(self):
        """
        Analyze and print missing values in the dataset.
        """
        missing_values = self.data.isnull().sum()
        missing_percentages = 100 * missing_values / len(self.data)
        missing_table = pd.concat([missing_values, missing_percentages], axis=1, keys=['Missing Values', 'Percentage'])
        print("\nMissing Value Analysis:")
        print(missing_table[missing_table['Missing Values'] > 0])
    
    def duplicate_rows_analysis(self):
        """
        Analyze and print the number of duplicate rows in the dataset.
        """
        duplicates = self.data.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates}")

    def advanced_correlation_analysis(self):
        """
        Analyze and visualize the correlation between numeric columns.
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(self.data[numeric_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()
        
    def distribution_plots(self):
        """
        Plot distribution and boxplots for numerical columns.
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

    def bar_plots(self):
        """
        Plot bar charts for categorical columns.
        """
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            plt.figure(figsize=(10, 6))
            self.data[col].value_counts().plot(kind='bar')
            plt.title(f"Value Counts for {col}")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.show()

    def skewness_analysis(self):
        """
        Analyze and print the skewness of numeric columns.
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        print("\nSkewness Analysis:")
        skewness = self.data[numeric_columns].apply(lambda x: stats.skew(x.dropna()))
        print(skewness)

    def unique_values_count(self):
        """
        Print the number of unique values for each column.
        """
        print("\nUnique Values Count:")
        print(self.data.nunique())

    def outlier_analysis(self):
        """
        Analyze and visualize outliers in the dataset.
        """
        print("\nOutlier Analysis:")
        for col in self.data.select_dtypes(include=[np.number]):
            outliers = self.data[(np.abs(stats.zscore(self.data[col])) > 3)]
            print(f"{col} has {len(outliers)} outliers")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col])
            plt.title(f"Outliers in {col}")
            plt.show()

    def target_variable_analysis(self, target_column):
        """
        Analyze and visualize the target variable.
        """
        if target_column in self.data.columns:
            print(f"\nAnalysis of Target Variable: {target_column}")
            
            # Plotting the distribution of the target variable
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[target_column], kde=True)
            plt.title(f"Distribution of {target_column}")
            plt.show()
            
            # Correlation with other features
            print(f"\nCorrelation with {target_column}:")
            corr_with_target = self.data.corr()[target_column].sort_values(ascending=False)
            print(corr_with_target)

    def interaction_analysis(self):
        """
        Analyze interactions between features in the dataset.
        """
        print("\nInteraction Analysis:")
        sns.pairplot(self.data.select_dtypes(include=[np.number]))
        plt.show()


    def time_series_analysis(self, time_column):
        """
        Analyze and visualize time-series data in the dataset.
        """
        if time_column in self.data.columns:
            print(f"\nTime-Series Analysis for {time_column}:")
            self.data[time_column] = pd.to_datetime(self.data[time_column])
            self.data.set_index(time_column, inplace=True)
            plt.figure(figsize=(10, 6))
            self.data.plot()
            plt.title(f"Time Series Plot for {time_column}")
            plt.show()

    def feature_importance_analysis(self, target_column):
        """
        Analyze and visualize feature importance using a simple model.
        """
        if target_column in self.data.columns:
            print(f"\nFeature Importance Analysis:")
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            if y.nunique() > 10:
                model = RandomForestRegressor()
            else:
                model = RandomForestClassifier()
            model.fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = X.columns
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance")
            plt.bar(range(X.shape[1]), importances[indices], align="center")
            plt.xticks(range(X.shape[1]), features[indices], rotation=90)
            plt.tight_layout()
            plt.show()

    def generate_report(self, output_path="report.html"):
        """
        Generate an automated report of the dataset using pandas_profiling.
        """
        profile = ProfileReport(self.data, title="Data Analysis Report", explorative=True)
        profile.to_file(output_path)


    def basic_analysis(self, target_column=None, time_column=None):
        """
        Perform a comprehensive basic analysis of the dataset.
        """
        self.memory_usage_analysis()
        self.basic_dataset_info()
        self.summary_statistics()
        self.data_type_analysis()
        self.missing_value_analysis()
        self.duplicate_rows_analysis()
        self.advanced_correlation_analysis()
        self.distribution_plots()
        self.bar_plots()
        self.skewness_analysis()
        self.unique_values_count()
        self.outlier_analysis()
        self.interaction_analysis()
        
        if target_column:
            self.target_variable_analysis(target_column)
        
        if time_column:
            self.time_series_analysis(time_column)
        
        # Optional: Generate a detailed report
        self.generate_report()

    def drop_columns_with_missing_values(self, df, drop_threshold):
        """
        Drop columns with missing values above the threshold.
        """
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index
        if len(cols_to_drop) > 0:
            print(f"Dropping columns with more than {drop_threshold*100}% missing values: {list(cols_to_drop)}")
        return df.drop(columns=cols_to_drop)

    def handle_missing_values(self, df, strategy='mean', fill_value=None):
        if df.empty:
            print("The DataFrame is empty. No missing values to handle.")
            return df

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        if numeric_columns.empty and categorical_columns.empty:
            print("No numeric or categorical columns to handle.")
            return df

        if strategy == 'drop':
            print("Dropping rows with missing values.")
            return df.dropna()
        
        if strategy in ['mean', 'median', 'most_frequent']:
            print(f"Imputing missing values using {strategy} strategy for numeric columns.")
            num_imputer = SimpleImputer(strategy=strategy)
            if not numeric_columns.empty:
                df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
            
            print("Imputing missing values using 'most_frequent' strategy for categorical columns.")
            cat_imputer = SimpleImputer(strategy='most_frequent')
            if not categorical_columns.empty:
                df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
        
        elif strategy == 'ffill':
            print("Forward filling missing values.")
            df = df.fillna(method='ffill')

        elif strategy == 'bfill':
            print("Backward filling missing values.")
            df = df.fillna(method='bfill')

        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError("For 'constant' strategy, fill_value must be provided.")
            print(f"Filling missing values with custom value: {fill_value}")
            df = df.fillna(value=fill_value)

        elif strategy == 'knn':
            print("Imputing missing values using KNN strategy.")
            imputer = KNNImputer(n_neighbors=5)
            if not numeric_columns.empty:
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        elif strategy == 'mice':
            print("Imputing missing values using MICE strategy.")
            imputer = IterativeImputer(random_state=0)
            if not numeric_columns.empty:
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        return df

    def handle_outliers(self, df, method, contamination):
        if df.empty:
            print("The DataFrame is empty. No outliers to handle.")
            return df
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if numeric_columns.empty:
            print("No numeric columns to handle outliers.")
            return df

        if method == 'iqr':
            print("Handling outliers using IQR method.")
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            print("Handling outliers using Z-score method.")
            df = df[(np.abs(stats.zscore(df[numeric_columns])) < 3).all(axis=1)]
        
        elif method == 'isolation_forest':
            print(f"Handling outliers using Isolation Forest with contamination={contamination}.")
            try:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outliers = iso_forest.fit_predict(df[numeric_columns])
                df = df[outliers != -1]
            except ValueError as e:
                print(f"Isolation Forest failed: {e}. Try adjusting the contamination parameter.")
        
        elif method == 'winsorize':
            print("Handling outliers using Winsorization.")
            for col in numeric_columns:
                df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])
        
        return df

    def remove_duplicate_rows(self, df):
        """
        Remove duplicate rows from the dataset.
        """
        print("Removing duplicate rows.")
        return df.drop_duplicates()

    def convert_date_columns(self, df, date_format):
        """
        Convert columns to datetime format.
        """
        if date_format:
            print(f"Converting date columns using format: {date_format}.")
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], format=date_format)
                    except ValueError:
                        pass  # Not a date column, skip
        return df

    def strip_whitespace(self, df):
        """
        Strip whitespace from string columns.
        """
        print("Stripping whitespace from string columns.")
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        return df

    def replace_infinity(self, df):
        """
        Replace infinity values with NaN.
        """
        print("Replacing infinity values with NaN.")
        return df.replace([np.inf, -np.inf], np.nan)

    def final_remove_nan(self, df):
        """
        Drop any remaining NaN values.
        """
        print("Dropping any remaining NaN values.")
        return df.dropna()

    def clean(self, missing='mean', outliers='iqr', drop_threshold=0.7, date_format=None, 
              remove_duplicates=True, iso_forest_contamination=0.1, fill_value=None):
        """
        Clean the dataset by handling missing values, outliers, duplicates, and performing basic cleaning tasks.
        
        Parameters:
        missing (str): Method to handle missing values. Method to handle missing values. Options: 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant', 'knn', 'mice', 'drop'
        outliers (str): Method to handle outliers. Options: 'iqr', 'zscore', 'isolation_forest', 'winsorize', None
        drop_threshold (float): Threshold for dropping columns with too many missing values (0.0 to 1.0)
        date_format (str): Format string for parsing dates (e.g., '%Y-%m-%d')
        remove_duplicates (bool): Whether to remove duplicate rows
        iso_forest_contamination (float): Contamination factor for Isolation Forest method.
        fill_value (any): Custom value for 'constant' strategy.

        Returns:
        pandas.DataFrame: Cleaned dataset
        """
        df = self.data.copy()

        # Start cleaning process
        print("Starting data cleaning process...")

        if df.empty:
            print("The DataFrame is empty. No cleaning to perform.")
            return df

        df = self.drop_columns_with_missing_values(df, drop_threshold)
        df = self.handle_missing_values(df, strategy=missing, fill_value=fill_value)
        df = self.handle_outliers(df, outliers, iso_forest_contamination)
        if remove_duplicates:
            df = self.remove_duplicate_rows(df)
        df = self.convert_date_columns(df, date_format)
        df = self.strip_whitespace(df)
        df = self.replace_infinity(df)
        df = self.final_remove_nan(df)

        # End cleaning process
        print("Data cleaning process completed.")
        
        self.data = df
        return df


    def fe(self, target_column=None, polynomial_degree=2, interaction_only=False, 
       bin_numeric=False, num_bins=5, cyclical_features=None, 
       text_columns=None, date_columns=None, math_transformations=None):
        """
        Perform extensive feature engineering on the dataset.

        Parameters:
        target_column (str): Name of the target column for feature selection
        polynomial_degree (int): Degree for polynomial feature generation
        interaction_only (bool): If True, only interaction features are produced
        bin_numeric (bool): If True, bin numeric features
        num_bins (int): Number of bins for numeric binning
        cyclical_features (list): List of cyclical features to encode
        text_columns (list): List of text columns for text feature extraction
        date_columns (list): List of date columns for date feature extraction
        math_transformations (dict): Dictionary of mathematical transformations to apply to numeric features

        Returns:
        pandas.DataFrame: Dataset with engineered features
        """
        df = self.data.copy()
        print("Starting feature engineering process...")

        # Numeric feature engineering
        df = self.create_polynomial_features(df, polynomial_degree, interaction_only)
        if bin_numeric:
            df = self.bin_numeric_features(df, num_bins)

        # Apply mathematical transformations
        if math_transformations:
            df = self.apply_math_transformations(df, math_transformations)

        # Categorical feature engineering
        df = self.create_categorical_interaction_features(df)

        # Cyclical feature encoding
        if cyclical_features:
            df = self.encode_cyclical_features(df, cyclical_features)

        # Text feature extraction
        if text_columns:
            df = self.extract_text_features(df, text_columns)

        # Date feature extraction
        if date_columns:
            df = self.extract_date_features(df, date_columns)

        # Feature selection based on correlation with target
        if target_column:
            df = self.select_features(df, target_column)

        print("Feature engineering process completed.")
        self.data = df
        return df

    def create_polynomial_features(self, df, degree, interaction_only):
        print(f"Creating polynomial features of degree {degree}")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(df[numeric_columns])
        poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(numeric_columns))
        return pd.concat([df, poly_features_df], axis=1)

    def bin_numeric_features(self, df, num_bins):
        print(f"Binning numeric features into {num_bins} bins")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[f'{col}_binned'] = pd.cut(df[col], bins=num_bins, labels=False)
        return df

    def create_categorical_interaction_features(self, df):
        print("Creating interaction features for categorical variables")
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for i in range(len(categorical_columns)):
            for j in range(i+1, len(categorical_columns)):
                col1, col2 = categorical_columns[i], categorical_columns[j]
                df[f'{col1}_{col2}_interaction'] = df[col1].astype(str) + '_' + df[col2].astype(str)
        return df

    def encode_cyclical_features(self, df, cyclical_features):
        print("Encoding cyclical features")
        for feature in cyclical_features:
            if feature in df.columns:
                max_value = df[feature].max()
                df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature]/max_value)
                df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature]/max_value)
        return df

    def extract_text_features(self, df, text_columns):
        print("Extracting features from text columns")
        for col in text_columns:
            if col in df.columns:
                df[f'{col}_length'] = df[col].str.len()
                df[f'{col}_word_count'] = df[col].str.split().str.len()
                df[f'{col}_avg_word_length'] = df[col].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if x else 0)
        return df

    def extract_date_features(self, df, date_columns):
        print("Extracting features from date columns")
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
        return df

    def select_features(self, df, target_column):
        print("Selecting features based on correlation with target")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        
        # Select top 20 features or features with MI score > 0.05, whichever is larger
        selected_features = mi_scores[mi_scores > 0.05].index.tolist()
        if len(selected_features) < 20:
            selected_features = mi_scores.nlargest(20).index.tolist()
        
        selected_features.append(target_column)
        return df[selected_features]

    def apply_math_transformations(self, df, transformations=None):
        """
        Apply mathematical transformations to numerical features.

        Parameters:
        df (pandas.DataFrame): The DataFrame to apply transformations to.
        transformations (dict): A dictionary where keys are column names and values are lists of transformations to apply.
                                Supported transformations: 'log', 'sqrt', 'reciprocal'.

        Returns:
        pandas.DataFrame: DataFrame with transformed features.
        """
        if transformations is None:
            transformations = {}

        print("Applying mathematical transformations to numeric features.")
        
        for col, trans_list in transformations.items():
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                for trans in trans_list:
                    if trans == 'log':
                        df[f'{col}_log'] = np.log1p(df[col])  # log1p is used to handle zero and negative values
                    elif trans == 'sqrt':
                        df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))  # clip to avoid sqrt of negative numbers
                    elif trans == 'reciprocal':
                        df[f'{col}_reciprocal'] = 1 / df[col].replace(0, np.nan)  # Replace zero to avoid division by zero

        return df
   


    def encode(self, methods=None, target_column=None):
        """
        Encode categorical variables using various encoding techniques.

        Parameters:
        methods (dict): A dictionary where keys are column names and values are encoding methods.
                        Supported methods: 'onehot', 'label', 'ordinal', 'target', 'woe', 'james_stein', 'catboost',  'binary'
        target_column (str): Name of the target column for supervised encoding methods

        Returns:
        pandas.DataFrame: DataFrame with encoded features
        """
        df = self.data.copy()
        print("Starting encoding process...")

        if methods is None:
            methods = {col: 'onehot' for col in df.select_dtypes(include=['object', 'category']).columns}

        for col, method in methods.items():
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in the dataset. Skipping.")
                continue

            if method == 'onehot':
                df = self._onehot_encode(df, col)
            elif method == 'label':
                df = self._label_encode(df, col)
            elif method == 'ordinal':
                df = self._ordinal_encode(df, col)
            elif method == 'target':
                if target_column is None:
                    raise ValueError("Target column must be specified for target encoding.")
                df = self._target_encode(df, col, target_column)
            elif method == 'woe':
                if target_column is None:
                    raise ValueError("Target column must be specified for WOE encoding.")
                df = self._woe_encode(df, col, target_column)
            elif method == 'james_stein':
                if target_column is None:
                    raise ValueError("Target column must be specified for James-Stein encoding.")
                df = self._james_stein_encode(df, col, target_column)
            elif method == 'catboost':
                if target_column is None:
                    raise ValueError("Target column must be specified for CatBoost encoding.")
                df = self._catboost_encode(df, col, target_column)
            elif method == 'binary':
                df = self._binary_encode(df, col)
            else:
                print(f"Warning: Unknown encoding method '{method}' for column '{col}'. Skipping.")

        print("Encoding process completed.")
        self.data = df
        return df

    def _onehot_encode(self, df, column):
        print(f"Applying One-Hot encoding to column: {column}")
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
        return pd.concat([df.drop(columns=[column]), encoded_df], axis=1)

    def _label_encode(self, df, column):
        print(f"Applying Label encoding to column: {column}")
        encoder = LabelEncoder()
        df[f"{column}_encoded"] = encoder.fit_transform(df[column])
        return df

    def _ordinal_encode(self, df, column):
        print(f"Applying Ordinal encoding to column: {column}")
        encoder = OrdinalEncoder()
        df[f"{column}_encoded"] = encoder.fit_transform(df[[column]])
        return df

    def _target_encode(self, df, column, target_column):
        print(f"Applying Target encoding to column: {column}")
        encoder = TargetEncoder()
        df[f"{column}_target_encoded"] = encoder.fit_transform(df[column], df[target_column])
        return df

    def _woe_encode(self, df, column, target_column):
        print(f"Applying Weight of Evidence encoding to column: {column}")
        encoder = WOEEncoder()
        df[f"{column}_woe_encoded"] = encoder.fit_transform(df[column], df[target_column])
        return df

    def _james_stein_encode(self, df, column, target_column):
        print(f"Applying James-Stein encoding to column: {column}")
        encoder = JamesSteinEncoder()
        df[f"{column}_js_encoded"] = encoder.fit_transform(df[column], df[target_column])
        return df

    def _catboost_encode(self, df, column, target_column):
        print(f"Applying CatBoost encoding to column: {column}")
        encoder = CatBoostEncoder()
        df[f"{column}_catboost_encoded"] = encoder.fit_transform(df[column], df[target_column])
        return df

    def _binary_encode(self, df, column):
        print(f"Applying Binary encoding to column: {column}")
        encoder = BinaryEncoder()
        binary_encoded = encoder.fit_transform(df[column])
        binary_encoded.columns = [f"{column}_bin_{i}" for i in range(len(binary_encoded.columns))]
        return pd.concat([df, binary_encoded], axis=1)

    def scale(self, method='standard', columns=None):
        """
        Scale numerical features in the dataset.

        Parameters:
        method (str): Scaling method to use. Options: 'standard', 'minmax', 'robust'.
        columns (list): List of columns to scale. If None, scales all numeric columns.

        Returns:
        pandas.DataFrame: DataFrame with scaled features.
        """
        df = self.data.copy()
        if df.empty:
            print("The DataFrame is empty. No scaling to perform.")
            return df

        df = self.replace_infinity(df)  # Replace infinity values with NaN before scaling
        df = self.handle_missing_values(df, strategy='mean')  # Handle any remaining NaN values

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            print(f"Applying Standard scaling to columns: {columns}")
            scaler = StandardScaler()
        elif method == 'minmax':
            print(f"Applying Min-Max scaling to columns: {columns}")
            scaler = MinMaxScaler()
        elif method == 'robust':
            print(f"Applying Robust scaling to columns: {columns}")
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method '{method}'.")

        if not columns.empty:
            df[columns] = scaler.fit_transform(df[columns])

        print("Scaling process completed.")
        self.data = df
        return df

    def normalize(self, method='l2', columns=None):
        """
        Normalize numerical features in the dataset.

        Parameters:
        method (str): Normalization method to use. Options: 'l1', 'l2', 'max'.
        columns (list): List of columns to normalize. If None, normalizes all numeric columns.

        Returns:
        pandas.DataFrame: DataFrame with normalized features.
        """
        df = self.data.copy()
        print(f"Starting normalization process using {method} normalization.")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        normalizer = Normalizer(norm=method)

        df[columns] = normalizer.fit_transform(df[columns])

        print("Normalization process completed.")
        self.data = df
        return df
    
    def pca_reduction(self, n_components=2):
        """
        Apply Principal Component Analysis (PCA) to reduce dimensionality.

        Parameters:
        n_components (int): Number of principal components to keep.

        Returns:
        pandas.DataFrame: DataFrame with principal components.
        """
        df = self.data.select_dtypes(include=[np.number]).copy()  # Only numeric data
        
        if df.isnull().any().any():
            print("Missing values detected. Handling missing values before applying PCA.")
            df = self.handle_missing_values(df, strategy='mean')

        if df.empty or df.shape[1] < n_components:
            raise ValueError("Insufficient data to perform PCA.")

        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(df)
        
        pca_df = pd.DataFrame(data=pca_components, columns=[f'PC{i+1}' for i in range(n_components)])
        
        print(f"PCA completed with {n_components} components.")
        
        return pd.concat([self.data.drop(columns=df.columns), pca_df], axis=1)

    def tsne_reduction(self, n_components=2, perplexity=30.0, n_iter=1000):
        """
        Apply t-SNE to reduce dimensionality.

        Parameters:
        n_components (int): Number of dimensions to reduce to (usually 2 or 3).
        perplexity (float): Perplexity parameter for t-SNE.
        n_iter (int): Number of iterations for optimization.

        Returns:
        pandas.DataFrame: DataFrame with t-SNE components.
        """
        df = self.data.select_dtypes(include=[np.number]).copy()  # Only numeric data
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        tsne_components = tsne.fit_transform(df)
        
        # Create a DataFrame with the t-SNE components
        tsne_df = pd.DataFrame(data=tsne_components, columns=[f't-SNE{i+1}' for i in range(n_components)])
        
        print(f"t-SNE completed with {n_components} components.")
        
        return pd.concat([self.data.drop(columns=df.columns), tsne_df], axis=1)

    def umap_reduction(self, n_components=2, n_neighbors=15, min_dist=0.1):
        """
        Apply UMAP to reduce dimensionality.

        Parameters:
        n_components (int): Number of dimensions to reduce to (usually 2 or 3).
        n_neighbors (int): Number of neighbors considered for UMAP.
        min_dist (float): Minimum distance between points in the low-dimensional space.

        Returns:
        pandas.DataFrame: DataFrame with UMAP components.
        """
        df = self.data.select_dtypes(include=[np.number]).copy()  # Only numeric data
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_components = umap_model.fit_transform(df)
        
        # Create a DataFrame with the UMAP components
        umap_df = pd.DataFrame(data=umap_components, columns=[f'UMAP{i+1}' for i in range(n_components)])
        
        print(f"UMAP completed with {n_components} components.")
        
        return pd.concat([self.data.drop(columns=df.columns), umap_df], axis=1)

    def dimreduction(self, method='pca', n_components=5):
        """
        Perform dimensionality reduction on the dataset.

        Parameters:
        method (str): The dimensionality reduction method. Options: 'pca', 'tsne', 'umap'
        n_components (int): Number of components to keep

        Returns:
        pandas.DataFrame: Dataset with reduced dimensions
        """
        df = self.data.copy()

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if method == 'pca':
            print(f"Applying PCA with {n_components} components.")
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(df[numeric_columns])
        elif method == 'tsne':
            print(f"Applying t-SNE with {n_components} components.")
            tsne = TSNE(n_components=n_components)
            reduced_data = tsne.fit_transform(df[numeric_columns])
        elif method == 'umap':
            print(f"Applying UMAP with {n_components} components.")
            umap_reducer = umap.UMAP(n_components=n_components)
            reduced_data = umap_reducer.fit_transform(df[numeric_columns])
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        reduced_df = pd.DataFrame(reduced_data, columns=[f'{method}_component_{i+1}' for i in range(n_components)])
        return pd.concat([df, reduced_df], axis=1)

    def correlation_selector(self, target_column, threshold=0.8):
        """
        Select features based on correlation with the target and remove highly correlated features.

        Parameters:
        target_column (str): Target column for correlation analysis.
        threshold (float): Correlation threshold for feature selection.

        Returns:
        pandas.DataFrame: DataFrame with selected features.
        """
        df = self.data.copy()
        corr_matrix = df.corr()
        target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
        
        # Select features with correlation above the threshold with the target
        selected_features = target_corr[target_corr > threshold].index.tolist()
        
        # Remove features that are highly correlated with each other
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                if abs(corr_matrix.loc[selected_features[i], selected_features[j]]) > threshold:
                    selected_features.remove(selected_features[j])
        
        print(f"Selected features based on correlation with threshold {threshold}.")
        return df[selected_features]
    
    def mutual_info_selector(self, target_column, n_features=10):
        """
        Select features based on mutual information with the target variable.

        Parameters:
        target_column (str): Target column for mutual information analysis.
        n_features (int): Number of top features to select.

        Returns:
        pandas.DataFrame: DataFrame with selected features.
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        selected_features = mi_scores.head(n_features).index.tolist()
        
        print(f"Selected top {n_features} features based on mutual information.")
        return self.data[selected_features + [target_column]]

    def variance_threshold_selector(self, threshold=0.01):
        """
        Select features based on variance threshold.

        Parameters:
        threshold (float): Variance threshold for feature selection.

        Returns:
        pandas.DataFrame: DataFrame with selected features.
        """
        df = self.data.select_dtypes(include=[np.number]).copy()  # Only numeric data
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        selected_columns = df.columns[selector.get_support()]
        
        print(f"Selected features with variance above {threshold}.")
        return self.data[selected_columns]
    
    def model_based_selector(self, target_column, model_type='random_forest', 
                         n_features=None, alpha=1.0, random_state=42):
        """
        Select features based on model feature importance or coefficients.

        Parameters:
        target_column (str): Target column for feature selection.
        model_type (str): Type of model to use for feature selection.
                        Options: 'random_forest', 'lasso', 'ridge'
        n_features (int): Number of top features to select. If None, selects all above a threshold.
        alpha (float): Regularization strength for Lasso or Ridge (ignored for random_forest).
        random_state (int): Random state for model reproducibility.

        Returns:
        pandas.DataFrame: DataFrame with selected features.
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        if model_type == 'random_forest':
            # Choose the right model based on the target type
            if y.dtype == 'object' or y.dtype.name == 'category':
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha, random_state=random_state)
        
        elif model_type == 'ridge':
            model = Ridge(alpha=alpha, random_state=random_state)
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported types: 'random_forest', 'lasso', 'ridge'")
        
        # Fit the model
        model.fit(X, y)
        
        if model_type == 'random_forest':
            # Use feature importances for RandomForest
            selector = SelectFromModel(model, prefit=True, max_features=n_features)
        else:
            # Use coefficients for Lasso or Ridge
            selector = SelectFromModel(model, prefit=True)
        
        selected_features = X.columns[selector.get_support()]
        
        print(f"Selected features based on {model_type} model.")
        
        return self.data[selected_features.tolist() + [target_column]]

    def feature_selection(self, target_column, method='correlation', n_features=None, 
                      correlation_threshold=0.8, model_type='random_forest', alpha=1.0, 
                      variance_threshold=0.0, random_state=42):
        """
        Unified method for feature selection using various techniques.

        Parameters:
        target_column (str): The name of the target column for feature selection.
        method (str): Feature selection method to use.
                    Options: 'correlation', 'mutual_info', 'variance', 'model_based'
        n_features (int): Number of top features to select (if applicable).
        correlation_threshold (float): Threshold for correlation-based feature selection.
        model_type (str): Model to use for model-based feature selection ('random_forest', 'lasso', 'ridge').
        alpha (float): Regularization strength for Lasso or Ridge (only used if model_type is 'lasso' or 'ridge').
        variance_threshold (float): Threshold for variance-based feature selection.
        random_state (int): Random state for reproducibility in model-based methods.

        Returns:
        pandas.DataFrame: DataFrame with selected features.
        """
        if method == 'correlation':
            return self.correlation_based_selector(target_column, threshold=correlation_threshold)
        
        elif method == 'mutual_info':
            return self.mutual_info_selector(target_column, n_features=n_features)
        
        elif method == 'variance':
            return self.variance_threshold_selector(target_column, threshold=variance_threshold)
        
        elif method == 'model_based':
            return self.model_based_selector(target_column, model_type=model_type, 
                                            n_features=n_features, alpha=alpha, 
                                            random_state=random_state)
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: 'correlation', 'mutual_info', 'variance', 'model_based'.")



    def balance_data(self, target_column, method='smote', sampling_strategy='auto', random_state=42):
        """
        Handle imbalanced data using various techniques.

        Parameters:
        target_column (str): The name of the target column.
        method (str): The balancing technique to use.
                    Options: 'random_over', 'smote', 'random_under', 'tomek_links'
        sampling_strategy (str or dict): Sampling strategy for the balancing technique.
                                        Default is 'auto'. For SMOTE, you can specify a dict to over-sample specific classes.
        random_state (int): Random state for reproducibility.

        Returns:
        pandas.DataFrame: Balanced DataFrame.
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        if method == 'random_over':
            return self.random_over_sampler(X, y, sampling_strategy, random_state)
        
        elif method == 'smote':
            return self.smote_sampler(X, y, sampling_strategy, random_state)
        
        elif method == 'random_under':
            return self.random_under_sampler(X, y, sampling_strategy, random_state)
        
        elif method == 'tomek_links':
            return self.tomek_links_sampler(X, y)
        
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: 'random_over', 'smote', 'random_under', 'tomek_links'.")

    def random_over_sampler(self, X, y, sampling_strategy, random_state):
        print("\nApplying Random Over-Sampling.")
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X, y)
        return pd.concat([X_res, y_res], axis=1)
    
    def smote_sampler(self, X, y, sampling_strategy, random_state):
        print("\nApplying SMOTE.")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        return pd.concat([X_res, y_res], axis=1)
    
    def random_under_sampler(self, X, y, sampling_strategy, random_state):
        print("\nApplying Random Under-Sampling.")
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_res, y_res = rus.fit_resample(X, y)
        return pd.concat([X_res, y_res], axis=1)
    
    def tomek_links_sampler(self, X, y):
        print("\nApplying Tomek Links.")
        tl = TomekLinks()
        X_res, y_res = tl.fit_resample(X, y)
        return pd.concat([X_res, y_res], axis=1)
    
    def resample_time_series(self, df, time_column, freq='D', method='mean'):
        """
        Resample the time series data.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        time_column (str): The name of the column containing the datetime information.
        freq (str): The frequency to resample the data to (e.g., 'D' for daily, 'M' for monthly).
        method (str): The method to apply for aggregation ('mean', 'sum', 'min', 'max').

        Returns:
        pandas.DataFrame: Resampled DataFrame.
        """
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)

        if method == 'mean':
            resampled_df = df.resample(freq).mean()
        elif method == 'sum':
            resampled_df = df.resample(freq).sum()
        elif method == 'min':
            resampled_df = df.resample(freq).min()
        elif method == 'max':
            resampled_df = df.resample(freq).max()
        else:
            raise ValueError(f"Unsupported resampling method: {method}")

        resampled_df = resampled_df.reset_index()
        return resampled_df

    def detrend_time_series(self, df, columns):
        """
        Remove the trend from the time series data.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        columns (list): List of column names to detrend.

        Returns:
        pandas.DataFrame: Detrended DataFrame.
        """
        for col in columns:
            print(f"\nDetrending column: {col}")
            df[f'{col}_detrended'] = detrend(df[col])
        return df

    def adjust_seasonality(self, df, columns, model='additive', period=None):
        """
        Adjust for seasonality in the time series data.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        columns (list): List of column names to adjust for seasonality.
        model (str): The type of seasonal adjustment ('additive' or 'multiplicative').
        period (int): The periodicity of the seasonality (e.g., 12 for monthly data).

        Returns:
        pandas.DataFrame: DataFrame with seasonality-adjusted features.
        """
        for col in columns:
            print(f"\nAdjusting seasonality for column: {col}")
            decomposition = seasonal_decompose(df[col], model=model, period=period)
            df[f'{col}_seasonal_adjusted'] = df[col] - decomposition.seasonal if model == 'additive' else df[col] / decomposition.seasonal
        return df

    def create_lag_features(self, df, columns, lags=[1, 2, 3]):
        """
        Create lag features for time series forecasting.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        columns (list): List of column names to create lag features for.
        lags (list): List of lag periods to create features for.

        Returns:
        pandas.DataFrame: DataFrame with lag features.
        """
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def time_series_preprocessing(self, time_column, freq='D', method='mean', detrend_columns=None, seasonality_columns=None,
                                  seasonality_model='additive', seasonality_period=None, lag_columns=None, lags=[1, 2, 3]):
        """
        Perform comprehensive time series preprocessing including resampling, detrending, seasonality adjustment, and lag feature creation.

        Parameters:
        time_column (str): The name of the column containing the datetime information.
        freq (str): The frequency to resample the data to.
        method (str): The method to apply for aggregation in resampling.
        detrend_columns (list): List of column names to detrend.
        seasonality_columns (list): List of column names to adjust for seasonality.
        seasonality_model (str): The type of seasonal adjustment ('additive' or 'multiplicative').
        seasonality_period (int): The periodicity of the seasonality.
        lag_columns (list): List of column names to create lag features for.
        lags (list): List of lag periods to create features for.

        Returns:
        pandas.DataFrame: Preprocessed DataFrame.
        """
        df = self.data.copy()

        print("\nResampling the time series data...")
        df = self.resample_time_series(df, time_column, freq, method)

        if detrend_columns:
            print("\nDetrending the time series data...")
            df = self.detrend_time_series(df, detrend_columns)

        if seasonality_columns:
            print("\nAdjusting for seasonality...")
            df = self.adjust_seasonality(df, seasonality_columns, model=seasonality_model, period=seasonality_period)

        if lag_columns:
            print("\nCreating lag features...")
            df = self.create_lag_features(df, lag_columns, lags)

        self.data = df
        return df
    
    def rolling_statistics(self, df, columns, window=3, statistics=['mean', 'std']):
        """
        Calculate rolling statistics for time series data.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        columns (list): List of column names to calculate rolling statistics for.
        window (int): The size of the rolling window.
        statistics (list): List of statistics to calculate ('mean', 'std', 'var', 'min', 'max').

        Returns:
        pandas.DataFrame: DataFrame with rolling statistics.
        """
        for col in columns:
            for stat in statistics:
                if stat == 'mean':
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                elif stat == 'std':
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                elif stat == 'var':
                    df[f'{col}_rolling_var_{window}'] = df[col].rolling(window=window).var()
                elif stat == 'min':
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                elif stat == 'max':
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                else:
                    raise ValueError(f"Unsupported rolling statistic: {stat}")
        return df

    def difference_time_series(self, df, columns, periods=1):
        """
        Apply differencing to make the time series data stationary.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        columns (list): List of column names to difference.
        periods (int): The number of periods to shift for differencing.

        Returns:
        pandas.DataFrame: Differenced DataFrame.
        """
        for col in columns:
            print(f"\nDifferencing column: {col}")
            df[f'{col}_diff_{periods}'] = df[col].diff(periods=periods)
        return df

    def fourier_transform(self, df, columns):
        """
        Apply Fourier Transform to extract frequency components from time series data.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        columns (list): List of column names to apply Fourier Transform on.

        Returns:
        pandas.DataFrame: DataFrame with Fourier components.
        """
        for col in columns:
            print(f"\nApplying Fourier Transform to column: {col}")
            fft_values = fft(df[col].dropna())
            df[f'{col}_fft_real'] = np.real(fft_values)
            df[f'{col}_fft_imag'] = np.imag(fft_values)
            df[f'{col}_fft_abs'] = np.abs(fft_values)
            df[f'{col}_fft_angle'] = np.angle(fft_values)
        return df

    def time_series_preprocessing(self, time_column, freq='D', method='mean', detrend_columns=None, seasonality_columns=None,
                                  seasonality_model='additive', seasonality_period=None, lag_columns=None, lags=[1, 2, 3],
                                  rolling_columns=None, rolling_window=3, rolling_stats=['mean', 'std'], difference_columns=None,
                                  difference_periods=1, fourier_columns=None):
        """
        Perform comprehensive time series preprocessing including resampling, detrending, seasonality adjustment, lag feature creation,
        rolling statistics, differencing, and Fourier Transform.

        Parameters:
        time_column (str): The name of the column containing the datetime information.
        freq (str): The frequency to resample the data to.
        method (str): The method to apply for aggregation in resampling.
        detrend_columns (list): List of column names to detrend.
        seasonality_columns (list): List of column names to adjust for seasonality.
        seasonality_model (str): The type of seasonal adjustment ('additive' or 'multiplicative').
        seasonality_period (int): The periodicity of the seasonality.
        lag_columns (list): List of column names to create lag features for.
        lags (list): List of lag periods to create features for.
        rolling_columns (list): List of column names to calculate rolling statistics for.
        rolling_window (int): The size of the rolling window.
        rolling_stats (list): List of statistics to calculate ('mean', 'std', 'var', 'min', 'max').
        difference_columns (list): List of column names to apply differencing.
        difference_periods (int): The number of periods to shift for differencing.
        fourier_columns (list): List of column names to apply Fourier Transform on.

        Returns:
        pandas.DataFrame: Preprocessed DataFrame.
        """
        df = self.data.copy()

        print("Resampling time series data")
        df = self.resample_time_series(df, time_column, freq, method)
        
        if detrend_columns:
            print("\nDetrending time series data")
            df = self.detrend_time_series(df, detrend_columns)
        
        if seasonality_columns:
            print("\nAdjusting for seasonality")
            df = self.adjust_seasonality(df, seasonality_columns, model=seasonality_model, period=seasonality_period)

        if lag_columns:
            print("\nCreating lag features")
            df = self.create_lag_features(df, lag_columns, lags)

        if rolling_columns:
            print("\nCalculating rolling statistics")
            df = self.rolling_statistics(df, rolling_columns, window=rolling_window, statistics=rolling_stats)

        if difference_columns:
            print("\nApplying differencing")
            df = self.difference_time_series(df, difference_columns, periods=difference_periods)

        if fourier_columns:
            print("\nApplying Fourier Transform")
            df = self.fourier_transform(df, fourier_columns)

        self.data = df
        return df
    
    def anonymize_data(self, columns, method='hash', mask_char='*', hash_algorithm='sha256'):
        """
        Anonymize data in specified columns using hashing or data masking.

        Parameters:
        columns (list): List of column names to anonymize.
        method (str): Method to use for anonymization ('hash' or 'mask').
        mask_char (str): Character to use for masking (if method is 'mask').
        hash_algorithm (str): Hashing algorithm to use ('md5', 'sha1', 'sha256', etc.).

        Returns:
        pandas.DataFrame: DataFrame with anonymized data.
        """
        df = self.data.copy()

        for col in columns:
            if col in df.columns:
                if method == 'hash':
                    df[col] = df[col].apply(lambda x: self._hash_value(str(x), hash_algorithm))
                elif method == 'mask':
                    df[col] = df[col].apply(lambda x: self._mask_value(str(x), mask_char))
                else:
                    raise ValueError(f"Unsupported anonymization method: {method}")

        self.data = df
        return df

    def _hash_value(self, value, algorithm):
        """
        Hash a given value using the specified algorithm.

        Parameters:
        value (str): The value to hash.
        algorithm (str): The hash algorithm to use ('md5', 'sha1', 'sha256', etc.).

        Returns:
        str: The hashed value.
        """
        try:
            hash_function = getattr(hashlib, algorithm)
            return hash_function(value.encode()).hexdigest()
        except AttributeError:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def _mask_value(self, value, mask_char='*'):
        """
        Mask a given value with a specified character.

        Parameters:
        value (str): The value to mask.
        mask_char (str): The character to use for masking.

        Returns:
        str: The masked value.
        """
        return mask_char * len(value)

    def randomize_characters(self, value, proportion=0.5):
        """
        Randomize characters in a string for anonymization.

        Parameters:
        value (str): The string to randomize.
        proportion (float): Proportion of characters to randomize (0.0 to 1.0).

        Returns:
        str: The string with randomized characters.
        """
        if not 0.0 <= proportion <= 1.0:
            raise ValueError("Proportion must be between 0.0 and 1.0")

        value = list(value)
        num_chars_to_randomize = int(len(value) * proportion)
        indices_to_randomize = random.sample(range(len(value)), num_chars_to_randomize)

        for i in indices_to_randomize:
            value[i] = random.choice(string.ascii_letters + string.digits)

        return ''.join(value)

    def apply_anonymization(self, columns, method='randomize', proportion=0.5, mask_char='*', hash_algorithm='sha256'):
        """
        Anonymize data using various techniques including randomization, masking, and hashing.

        Parameters:
        columns (list): List of columns to anonymize.
        method (str): Method to use ('randomize', 'mask', 'hash').
        proportion (float): Proportion of characters to randomize (for 'randomize' method).
        mask_char (str): Character to use for masking (for 'mask' method).
        hash_algorithm (str): Hashing algorithm to use (for 'hash' method).

        Returns:
        pandas.DataFrame: DataFrame with anonymized data.
        """
        df = self.data.copy()

        for col in columns:
            if method == 'randomize':
                df[col] = df[col].apply(lambda x: self.randomize_characters(str(x), proportion))
            elif method == 'mask':
                df[col] = df[col].apply(lambda x: self._mask_value(str(x), mask_char))
            elif method == 'hash':
                df[col] = df[col].apply(lambda x: self._hash_value(str(x), hash_algorithm))
            else:
                raise ValueError(f"Unsupported anonymization method: {method}")

        self.data = df
        return df

def load(filepath):
    return AutoDataPreprocess(filepath)