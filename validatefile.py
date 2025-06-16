import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings

class DataValidator:
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        if missing_cols:
            self.errors.append(f"Missing columns: {missing_cols}")
        
        if extra_cols:
            self.warnings.append(f"Extra columns found: {extra_cols}")
        
        return len(missing_cols) == 0
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        type_errors = []
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    type_errors.append(f"Column '{col}': expected {expected_type}, got {actual_type}")
        
        if type_errors:
            self.errors.extend(type_errors)
        
        return len(type_errors) == 0
    
    def validate_value_ranges(self, df: pd.DataFrame, range_checks: Dict[str, Dict]) -> bool:
        range_errors = []
        
        for col, checks in range_checks.items():
            if col in df.columns:
                if 'min' in checks:
                    invalid_count = (df[col] < checks['min']).sum()
                    if invalid_count > 0:
                        range_errors.append(f"Column '{col}': {invalid_count} values below minimum {checks['min']}")
                
                if 'max' in checks:
                    invalid_count = (df[col] > checks['max']).sum()
                    if invalid_count > 0:
                        range_errors.append(f"Column '{col}': {invalid_count} values above maximum {checks['max']}")
        
        if range_errors:
            self.errors.extend(range_errors)
        
        return len(range_errors) == 0
    
    def check_missing_values(self, df: pd.DataFrame, max_missing_ratio: float = 0.1) -> bool:
        missing_issues = []
        
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                missing_issues.append(f"Column '{col}': {missing_ratio:.2%} missing values")
        
        if missing_issues:
            self.warnings.extend(missing_issues)
        
        return len(missing_issues) == 0
    
    def generate_report(self) -> Dict[str, Any]:
        return {
            'validation_passed': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }