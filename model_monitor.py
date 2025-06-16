import time
import psutil
import logging
from datetime import datetime
import json
import pandas as pd

class ModelPerformanceMonitor:
    def __init__(self, log_file='model_performance.log'):
        self.log_file = log_file
        self.logger = self._setup_logger()
        self.metrics_history = []
    
    def _setup_logger(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def monitor_prediction_time(self, model, X_test):
        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        samples_per_second = len(X_test) / prediction_time
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'prediction_time': prediction_time,
            'samples_processed': len(X_test),
            'samples_per_second': samples_per_second,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent()
        }
        
        self.metrics_history.append(metrics)
        self.logger.info(f"Performance metrics: {json.dumps(metrics)}")
        
        return predictions, metrics
    
    def check_model_drift(self, reference_data, current_data, threshold=0.05):
        from scipy.stats import ks_2samp
        
        drift_detected = False
        drift_scores = {}
        
        for col in reference_data.columns:
            if col in current_data.columns:
                statistic, p_value = ks_2samp(reference_data[col], current_data[col])
                drift_scores[col] = {'statistic': statistic, 'p_value': p_value}
                
                if p_value < threshold:
                    drift_detected = True
                    self.logger.warning(f"Data drift detected in column {col}: p-value = {p_value}")
        
        return drift_detected, drift_scores
    
    def generate_performance_report(self):
        if not self.metrics_history:
            return "No performance data available"
        
        df = pd.DataFrame(self.metrics_history)
        
        report = {
            'total_predictions': df['samples_processed'].sum(),
            'avg_prediction_time': df['prediction_time'].mean(),
            'avg_samples_per_second': df['samples_per_second'].mean(),
            'max_memory_usage_mb': df['memory_usage_mb'].max(),
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'monitoring_period': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        return report