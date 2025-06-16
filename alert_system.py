import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from datetime import datetime

class AlertSystem:
    def __init__(self, smtp_server, smtp_port, email, password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'prediction_time': 2.0,
            'memory_usage': 1000,  # MB
            'error_rate': 0.01
        }
    
    def check_thresholds(self, metrics):
        alerts = []
        
        if 'accuracy' in metrics and metrics['accuracy'] < (1 - self.alert_thresholds['accuracy_drop']):
            alerts.append(f"Model accuracy dropped to {metrics['accuracy']:.3f}")
        
        if 'prediction_time' in metrics and metrics['prediction_time'] > self.alert_thresholds['prediction_time']:
            alerts.append(f"Prediction time exceeded threshold: {metrics['prediction_time']:.2f}s")
        
        if 'memory_usage_mb' in metrics and metrics['memory_usage_mb'] > self.alert_thresholds['memory_usage']:
            alerts.append(f"Memory usage high: {metrics['memory_usage_mb']:.1f}MB")
        
        return alerts
    
    def send_alert_email(self, alerts, recipient_emails):
        if not alerts:
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.email
        msg['To'] = ', '.join(recipient_emails)
        msg['Subject'] = f"Model Performance Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        body = "Model Performance Alerts:\n\n" + '\n'.join(f"• {alert}" for alert in alerts)
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()
            print("Alert email sent successfully")
        except Exception as e:
            print(f"Failed to send alert email: {e}")
    
    def log_alert(self, alerts, log_file='alerts.json'):
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts
        }
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(alert_entry) + '\n')
        except Exception as e:
            print(f"Failed to log alert: {e}")