# main.py
import serial
import time
import pandas as pd
from datetime import datetime
import os
from xai import CropXAI

def main():
    # Initialize system
    xai = CropXAI()
    if not xai.train():
        return
    
    try:
        # Serial connection
        ser = serial.Serial('COM8', 9600)
        ser.setDTR(False)
        time.sleep(1)
        ser.flushInput()
        ser.setDTR(True)
        
        # Data file setup
        csv_file = f"{datetime.now().strftime('%Y-%m-%d')}_data.csv"
        columns = ["Timestamp", "Soil(%)", "Temp(C)", "Humidity(%)", "Prediction"]
        
        if not os.path.exists(csv_file):
            pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
            
        while True:
            if ser.in_waiting > 0:
                raw = ser.readline().decode().strip()
                vals = raw.split(',')
                
                if len(vals) == 3 and xai.validate(*vals):
                    soil = float(vals[0])
                    temp = float(vals[1])
                    hum = float(vals[2])
                    
                    # Get prediction
                    result = xai.predict(soil, temp, hum)
                    
                    # Save data
                    new_row = [
                        datetime.now().strftime("%H:%M:%S"),
                        soil,
                        temp,
                        hum,
                        result['prediction']
                    ]
                    pd.DataFrame([new_row], columns=columns).to_csv(
                        csv_file, mode='a', header=False, index=False)
                    
                    # Display results
                    print(f"\n{'='*40}")
                    print(f"Recommended Crop: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2%}")
                    print("Feature Contributions:")
                    for feat, val in zip(result['explanation']['features'], 
                                      result['explanation']['values']):
                        print(f"  {feat}: {val:.2f}")
                    print(f"{'='*40}\n")
                    
    except KeyboardInterrupt:
        print("\nExiting...")
        ser.close()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()