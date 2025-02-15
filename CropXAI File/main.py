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
        # Open the serial port once
        ser = serial.Serial('COM8', 9600, timeout=1)
        ser.setDTR(False)
        time.sleep(1)
        ser.flushInput()
        ser.setDTR(True)
    except Exception as e:
        print("Failed to open serial port:", e)
        return

    # Define CSV file and columns once
    csv_file = f"{datetime.now().strftime('%Y-%m-%d')}_data.csv"
    columns = ["Timestamp", "Soil(%)", "Temp(C)", "Humidity(%)", "Prediction"]
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
    
    while True:
        try:
            # Read from serial
            s_bytes = ser.readline()
            if not s_bytes:
                time.sleep(0.1)
                continue  # No data received, try again
            
            raw = s_bytes.decode('utf-8', errors='ignore').strip()
            if not raw:
                continue

            vals = raw.split(',')
            if len(vals) == 3 and system.validate(*vals):
                soil, temp, hum = map(float, vals)
                
                # Get prediction and explanation
                result = system.predict(soil, temp, hum)
                
                # Save the entry to CSV
                pd.DataFrame([[datetime.now().strftime("%H:%M:%S"),
                               soil, temp, hum, result['prediction']]],
                             columns=columns).to_csv(csv_file, mode='a', header=False, index=False)
                
                # Display results
                print(f"\nInput: {soil}%, {temp}°C, {hum}%")
                print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
                key_factors = {feat: round(float(val), 4) for feat, val in 
                               zip(result['explanation']['features'], result['explanation']['values'])}
                print("Key Factors:", key_factors)
            else:
                # Optionally log if the received data is not valid
                print("Invalid data received:", raw)
                
        except KeyboardInterrupt:
            print("\nSystem shutdown initiated...")
            break
        except Exception as e:
            print(f"Critical error: {str(e)}")
    
    ser.close()

if __name__ == "__main__":
    main()
