import pandas as pd
import numpy as np
from scipy import stats

class DataQualityMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        completeness = {}
        for col in df.columns:
            completeness[col] = (df[col].count() / len(df)) * 100
        return completeness
    
    def calculate_uniqueness(self, df: pd.DataFrame) -> Dict[str, float]:
        uniqueness = {}
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                unique_ratio = (df[col].nunique() / df[col].count()) * 100
                uniqueness[col] = unique_ratio
        return uniqueness
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = outlier_count
        
        return outliers
    
    def calculate_distribution_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        distribution_metrics = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            metrics = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skewness': stats.skew(df[col].dropna()),
                'kurtosis': stats.kurtosis(df[col].dropna())
            }
            distribution_metrics[col] = metrics
        
        return distribution_metrics