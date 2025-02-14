# xai.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
        """Train model with SHAP integration"""
        try:
            X, y = self.load_data(data_path)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            # Initialize SHAP explainer
            self.explainer = shap.Explainer(
                self.model,
                X,
                feature_perturbation="interventional"
            )
            
            joblib.dump(self.model, 'crop_model.pkl')
            print(f"Trained with classes: {list(self.class_names)}")
            return True
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False
    
    def predict(self, soil, temp, hum):
        """Get prediction with explanation"""
        input_data = np.array([[soil, temp, hum]])
        
        # Get prediction
        encoded_pred = self.model.predict(input_data)[0]
        proba = self.model.predict_proba(input_data)[0]
        
        # Get SHAP values
        shap_values = self.explainer(input_data)
        
        return {
            'prediction': self.le.inverse_transform([encoded_pred])[0],
            'confidence': float(np.max(proba)),
            'probabilities': dict(zip(self.class_names, proba)),
            'explanation': {
                'features': self.feature_names,
                'values': list(shap_values.values[0, :, encoded_pred]),
                'base_value': float(shap_values.base_values[0, encoded_pred])
            }
        }

    def validate(self, soil, temp, hum):
        """Validate sensor inputs"""
        try:
            return all([
                0 <= float(soil) <= 1023,
                -40 <= float(temp) <= 80,
                0 <= float(hum) <= 100
            ])
        except ValueError:
            return False

if __name__ == "__main__":
    xai = CropXAI()
    if xai.train():
        result = xai.predict(70, 25, 85)
        print("Prediction:", result['prediction'])
        print("Feature Contributions:", result['explanation']['values'])