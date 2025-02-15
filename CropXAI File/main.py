# main.py
import serial
import time
import pandas as pd
from datetime import datetime
import os
from xai import CropXAI

def main():
    system = CropXAI()
    if not system.train():
        return
    
    try:
        # Serial connection with timeout
        with serial.Serial('COM8', 9600, timeout=1) as ser:
            ser.setDTR(False)
            time.sleep(1)
            ser.flushInput()
            ser.setDTR(True)
            
            # Daily data file
            csv_file = f"{datetime.now().strftime('%Y-%m-%d')}_data.csv"
            columns = ["Timestamp", "Soil(%)", "Temp(C)", "Humidity(%)", "Prediction"]
            
            # Initialize CSV with header
            if not os.path.exists(csv_file):
                pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
            
            if ser.in_waiting > 0:
                raw = ser.readline().decode().strip()
                if len((vals := raw.split(','))) == 3 and system.validate(*vals):
                    soil, temp, hum = map(float, vals)
                        
                    # Get prediction
                    result = system.predict(soil, temp, hum)
                        
                    # Save entry
                    pd.DataFrame([[
                        datetime.now().strftime("%H:%M:%S"),
                        soil,
                        temp,
                        hum,
                        result['prediction']
                    ]], columns=columns).to_csv(csv_file, mode='a', header=False, index=False)
                        
                    # Display results
                    print(f"\n{'='*40}")
                    print(f"Recommended: {result['prediction']} ({result['confidence']:.2%})")
                    print("Key Factors:")
                    for feat, val in zip(result['explanation']['features'], 
                                      result['explanation']['values']):
                        print(f"  {feat}: {val:+.2f}")
                    print(f"{'='*40}\n")
 
    except KeyboardInterrupt:
        print("\nSystem shutdown initiated...")
    except Exception as e:
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()
