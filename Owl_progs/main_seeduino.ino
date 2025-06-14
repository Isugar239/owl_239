#include <Keypad.h> // подключаем библиотеку Keypad
#include "Arduino.h"
#include <EEPROM.h>

static uint32_t time_search = millis();

int v = 80, condition = 0, time_cycle = 9500, fall_value [4] = {850, 950, 925, 970}, light_sens [4] = {0, 0, 0, 0};
bool turn = false;

void motor(int motorL, int motorR){
  digitalWrite(4, (motorL > 0));
  motorL = abs(motorL);
  analogWrite(5, min(255, motorL));

  digitalWrite(7, (motorR < 0));
  motorR = abs(motorR);
  analogWrite(6, min(255, motorR));
}
void alignment(){
  if (condition == 1){
    motor(-v, v);
  } else if (condition == 3){
    motor(v, -v);
  } else if (condition == 2){
    motor(0, 0);
  }
}
void searching(){
  static uint32_t timer = millis();
  if ((millis() - timer) % time_cycle < 700){
    motor(90, -90);
    turn = true;
  } else if ((millis() - timer) % time_cycle < 3000){
    motor(0, 0);
    turn = false;
  } else if ((millis() - timer) % time_cycle < 3700){
    motor(-73, -73);
    turn = false;
  } else if ((millis() - timer) % time_cycle < 5000){
    motor(0, 0);
    turn = false;
  } else if ((millis() - timer) % time_cycle < 5700){
    motor(-90, 90);
    turn = true;
  } else if ((millis() - timer) % time_cycle < 6900){
    motor(0, 0);
    turn = false;
  } else if ((millis() - timer) % time_cycle < 7300){
    motor(-70, -70);
    turn = false;
  } else{
    motor(0, 0);
    turn = false;
  }
}
void light_sensors_calibr(){
  if (condition == 7){
    delay(2000);
    sens_update();
    
    EEPROM.write(0, light_sens[0]);
    EEPROM.write(1, light_sens[1]);
    EEPROM.write(2, light_sens[2]);
    EEPROM.write(3, light_sens[3]);
    for (int sensor = 0; sensor < 4; sensor ++){
      fall_value[sensor] = EEPROM.read(sensor) + 15;
      Serial.print(fall_value[sensor]);
      Serial.print("\t");
    }
    Serial.println("");
  }
  condition = 0;
}
void sens_update(){
  for (int sensor = 0; sensor < 4; sensor ++){
    light_sens[sensor] = map(analogRead(sensor), 0, 1023, 0, 255);
    Serial.print(light_sens[sensor]);
    Serial.print("\t");
  }
  Serial.println("");
}
void falling_defence(){
  sens_update();
  if ((light_sens[0] > fall_value[0] || light_sens[1] > fall_value[1] || light_sens[2] > fall_value[2] || light_sens[3] > fall_value[3]) && !turn){
    motor(90, 87);
    delay(1000);
  } else if (condition == 0){
    motor(0, 0);
  }
}
void uart(){ 
  if (Serial2.available()){
    condition = int(Serial2.read()) - 48;
  }
  Serial.println(condition);
  Serial1.write(condition);
}

void setup() {
  pinMode(4, 1);
  pinMode(7, 1);
  Serial.begin(9600);
  Serial1.begin(9600);
  Serial2.begin(9600);
  Serial3.begin(9600);
  
  for (int sensor = 0; sensor < 4; sensor ++){
    fall_value[sensor] = EEPROM.read(sensor) + 15;
    Serial.print(fall_value[sensor]);
    Serial.print("\t");
  }
  Serial.println("");
  delay(2000);

  //ожидание концепции
  
}

void loop() {
  falling_defence();
  uart();
  if (condition == 6){
    //searching();
  } if (condition == 5){
    motor(0, 0);
  } if (condition == 4){
    motor(0, 0);
  } if (condition == 7){
    light_sensors_calibr();
  }else{
    alignment();
  }
  // put your main code here, to run repeatedly:

}
