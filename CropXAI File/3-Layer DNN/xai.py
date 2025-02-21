import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropXAI:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.feature_names = ["Soil Humidity(%)", "Temperature (C)", "Humidity(%)"]
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_data(self, data_path):
        """Load and preprocess agricultural sensor data"""
        df = pd.read_csv(data_path)
        X = df[self.feature_names]
        y = self.le.fit_transform(df['Crop'].str.strip().str.lower())
        self.class_names = self.le.classes_
        
        # Agricultural data augmentation
        X_augmented = pd.concat([
            X,
            X * np.random.uniform(0.95, 1.05, size=X.shape),  # Sensor calibration drift
            X + np.random.normal(0, 0.1, size=X.shape)        # Measurement noise
        ])
        y_augmented = np.concatenate([y, y, y])
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_augmented)
        return X_scaled, y_augmented
    
    def create_model(self, input_size, output_size):
        """Deep neural network for crop prediction"""
        return nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        ).to(self.device)
    
    def train(self, data_path='crop_training_data.csv'):
        """Train model with SHAP integration"""
        try:
            X, y = self.load_data(data_path)
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            self.model = self.create_model(X.shape[1], len(self.class_names))
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Agricultural training loop
            self.model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 50 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            
            # SHAP initialization with agricultural data
            self.model.cpu().eval()
            background = torch.FloatTensor(X[:100])  # First 100 samples
            self.explainer = shap.DeepExplainer(
                model=self.model,
                data=background
            )
            self.model.to(self.device)
            
            # Save agricultural model components
            torch.save(self.model.state_dict(), 'crop_model.pth')
            joblib.dump((self.scaler, self.le), 'preprocessors.pkl')
            print(f"Trained on {len(self.class_names)} crops")
            return True
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False
    
    def predict(self, soil, temp, hum):
        """Get agricultural prediction with explanations"""
        try:
            if not self.validate(soil, temp, hum):
                return {"error": "Invalid sensor values"}
            
            # Agricultural input preprocessing
            input_scaled = self.scaler.transform([[soil, temp, hum]])
            input_tensor = torch.FloatTensor(input_scaled).to(self.device)
            
            # Get prediction probabilities
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            encoded_pred = np.argmax(proba)
            
            # SHAP calculation for agricultural explainability
            self.model.cpu().eval()
            shap_values = self.explainer.shap_values(input_tensor.cpu())
            self.model.to(self.device)
            
            # Handle SHAP output for agricultural features
            if isinstance(shap_values, list):
                shap_array = np.stack(shap_values)
            else:
                # Ensure shape (n_classes, 1, n_features)
                shap_array = shap_values.reshape(len(self.class_names), 1, -1)
            
            return {
                'prediction': self.le.inverse_transform([encoded_pred])[0],
                'confidence': float(np.max(proba)),
                'probabilities': dict(zip(self.class_names, proba)),
                'explanation': {
                    'features': self.feature_names,
                    'values': list(shap_array[encoded_pred].flatten()),
                    'base_value': float(self.explainer.expected_value[encoded_pred])
                }
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def validate(self, soil, temp, hum):
        """Validate agricultural sensor ranges"""
        try:
            return all([
                0 <= float(soil) <= 1023,    # Capacitive soil moisture range
                -40 <= float(temp) <= 80,    # Agricultural temperature extremes
                0 <= float(hum) <= 100       # Relative humidity percentage
            ])
        except ValueError:
            return False

if __name__ == "__main__":
    xai = CropXAI()
    if xai.train():
        test_values = [(83, 24.5, 72), (73, 28.9, 71), (79, 25.2, 87)]
        for soil, temp, hum in test_values:
            result = xai.predict(soil, temp, hum)
            print(f"\nInput: {soil}%, {temp}°C, {hum}%")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Predicted Crop: {result['prediction']} ({result['confidence']:.2%} confidence)")
                print("Feature Contributions:")
                for feat, val in zip(result['explanation']['features'], result['explanation']['values']):
                    print(f"- {feat}: {val:.4f}")
