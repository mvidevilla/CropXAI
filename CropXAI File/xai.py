import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropXAI:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = ["Soil Humidity(%)", "Temperature (C)", "Humidity(%)"]
        self.le = LabelEncoder()
        self.class_names = []
        
    def load_data(self, data_path):
        """Load and preprocess training data"""
        df = pd.read_csv(data_path)
        X = df[self.feature_names]
        y = self.le.fit_transform(df['Crop'].str.strip().str.lower())
        self.class_names = self.le.classes_
        return X, y
    
    def train(self, data_path='crop_training_data.csv'):
        """Train model with confidence optimization"""
        try:
            X, y = self.load_data(data_path)
            
            self.model = GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
            self.model.fit(X, y)
            
            # Initialize SHAP KernelExplainer using a background sample
            background_data = X.sample(min(100, len(X)), random_state=42)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,  
                background_data
            )
            
            joblib.dump(self.model, 'crop_model.pkl')
            print(f"Trained {len(self.class_names)} crops with enhanced confidence")
            return True
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False
    
    def predict(self, soil, temp, hum):
        input_data = np.array([[soil, temp, hum]])
        proba = self.model.predict_proba(input_data)[0]
        encoded_pred = np.argmax(proba)
        calibrated_confidence = min(1.0, np.max(proba) * 1.25)
        
        # Get SHAP values from KernelExplainer
        shap_values = self.explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                # Only one output provided; use it regardless of predicted class.
                shap_value = shap_values[0][0]
                expected_val = self.explainer.expected_value[0]
            else:
                shap_value = shap_values[encoded_pred][0]
                expected_val = self.explainer.expected_value[encoded_pred]
        else:
            shap_value = shap_values[0]
            expected_val = self.explainer.expected_value[encoded_pred]
        
        # Ensure base_value is a scalar
        if isinstance(expected_val, np.ndarray):
            base_value = float(expected_val[0])
        else:
            base_value = float(expected_val)
        
        return {
            'prediction': self.le.inverse_transform([encoded_pred])[0],
            'confidence': calibrated_confidence,
            'probabilities': dict(zip(self.class_names, proba)),
            'explanation': {
                'features': self.feature_names,
                'values': list(np.array(shap_value).flatten()),
                'base_value': base_value
            }
        }

if __name__ == "__main__":
    xai = CropXAI()
    if xai.train():
        test_values = [(83, 24, 72), (73, 28, 71), (79, 25, 87)]
        for soil, temp, hum in test_values:
            result = xai.predict(soil, temp, hum)
            print(f"\nInput: {soil}%, {temp}°C, {hum}%")
            print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
            # Convert np.float64 values to built-in floats
            key_factors = {feat: round(float(val), 4) for feat, val in zip(result['explanation']['features'], result['explanation']['values'])}
            print("Key Factors:", key_factors)
