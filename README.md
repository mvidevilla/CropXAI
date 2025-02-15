Current version utilizes Arduino Uno connected to port COM8

Capacitive Soil Moisture Sensor V1.2 connected to Analog In 0

DHT22 Sensor connected to Digital Pin 2

(Schematics uploaded later)

================================================================

Uses Random Forest Model to create multiple sets of values and 
creates predictions based on their cumulative average

Uses SHAP to explain the deviation of input values from the
average calculated by the Random Forest Model

Existing Issue:
Low confidence level: 54% and below
