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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import os
from ydata_profiling import ProfileReport
from sqlalchemy import create_engine
import requests


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
        """
        Handle missing values in the dataset.
        """
        if strategy == 'drop':
            print("Dropping rows with missing values.")
            return df.dropna()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        if strategy in ['mean', 'median', 'mode']:
            print(f"Imputing missing values using {strategy} strategy for numeric columns.")
            num_imputer = SimpleImputer(strategy=strategy)
            df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
            
            print("Imputing missing values using 'most_frequent' strategy for categorical columns.")
            cat_imputer = SimpleImputer(strategy='most_frequent')
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
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        elif strategy == 'mice':
            print("Imputing missing values using MICE strategy.")
            imputer = IterativeImputer(random_state=0)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        return df

    def handle_outliers(self, df, method, contamination):
        """
        Handle outliers in the dataset.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
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
       text_columns=None, date_columns=None):
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

        Returns:
        pandas.DataFrame: Dataset with engineered features
        """
        df = self.data.copy()
        print("Starting feature engineering process...")

        # Numeric feature engineering
        df = self.create_polynomial_features(df, polynomial_degree, interaction_only)
        if bin_numeric:
            df = self.bin_numeric_features(df, num_bins)

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

    def encode(self, method='onehot'):
        # Implement encoding here
        pass

    def scale(self, method='standard'):
        # Implement scaling here
        pass

    def normalize(self, method='minmax'):
        # Implement normalization here
        pass

    def balance(self, method='smote'):
        # Implement balancing here
        pass

    def dimreduction(self, method='pca', n_components=5):
        # Implement dimensionality reduction here
        pass

def load(filepath):
    return AutoDataPreprocess(filepath)