import datetime

class HybridIDS:
    def __init__(self, supervised_model, unsupervised_model):
        self.supervised = supervised_model
        self.unsupervised = unsupervised_model

    def analyze(self, data_list):
        """
        True Hybrid Decision Chain:
        1. Signature Engine (RF): Known?
        2. If Confidence < 0.7 or 'Normal' -> Behavioral Audit (IF)
        3. If Audit finds Anomaly -> Flag as 'Suspicious Behavior/Unknown'
        """
        # Signature Engine (Random Forest)
        supervised_labels, confidences = self.supervised.predict(data_list, return_proba=True)
        
        # Behavioral Engine (Isolation Forest)
        unsupervised_results = self.unsupervised.detect(data_list)
        
        final_results = []
        for i in range(len(data_list)):
            s_label = supervised_labels[i]
            s_conf = confidences[i]
            u_is_anomaly = unsupervised_results[i]['prediction'] == "Anomaly"
            
            # Decision Chain Logic
            if s_label != "Normal" and s_conf > 0.7:
                # 1. Accepted Known Attack
                status = s_label
                detection_type = "Verified Signature"
                severity = "Critical" if "DoS" in s_label or "Mirai" in s_label else "High"
            else:
                # 2. Low Confidence or "Normal" -> Perform Behavioral Audit
                if u_is_anomaly:
                    status = "Unknown Anomaly"
                    detection_type = f"Behavioral Audit (Audit triggered: {s_label} @{round(s_conf*100)}%)"
                    severity = "Warning" if s_label == "Normal" else "High"
                else:
                    status = "Safe"
                    detection_type = "Verified Normal (Hybrid Audit)"
                    severity = "Safe"

            final_results.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "status": status,
                "confidence": round(s_conf * 100, 2),
                "detection_mode": detection_type,
                "severity": severity,
                "raw_info": data_list[i]
            })
            
        return final_results
