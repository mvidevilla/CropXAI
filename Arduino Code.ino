#include <DHT.h>
#include <Adafruit_Sensor.h>

#define DHTPIN 2
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

const int dry = 455;
const int wet = 210;

int sensorPin = A0;  
int sensorValue = analogRead(A0);  
int percent = 0;

void setup() {
  Serial.begin(9600);
  dht.begin();
}

void loop() {
  sensorValue = analogRead(sensorPin);
  percent = convertToPercent(sensorValue);
  
  printValuesToSerial();
  printHumidTemp();
  Serial.println();

  delay(2000);
}

int convertToPercent(int value){
  int percentValue = 0;
  percentValue = map(value, wet, dry, 100, 0);
  return percentValue;
}

void printValuesToSerial(){
  Serial.print(percent);
  Serial.print(",");
}

void printHumidTemp(){
  Serial.print(dht.readTemperature());
  Serial.print(",");
  Serial.print(dht.readHumidity());
}
