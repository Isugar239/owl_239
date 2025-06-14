int condition = 5, last_condition = 0, left_eye_now = 60, right_eye_now = 100, left_ear_now = 64, right_ear_now = 72, beak_now = 98, wink = 0;
int eyes_step = 5, ears_step = 2, beak_step = 1, radius_now = 20, time_tick = 15, angle = 180;
uint32_t detecting_sleeping = millis();
bool blink_update = true, first_part = true, loading_update = true;
uint32_t time_blinking = millis(), update_time = millis(), close_eyes = 100, time_search = millis();
uint32_t emotions_time = millis();
uint64_t timer = micros(), timer_waiting = micros();
float x = 0, y = 0, r = 40;

#include <Servo.h>
#include <SPI.h>
#include <Adafruit_GC9A01A.h>
#include <Adafruit_SPIDevice.h>

#define left_eye_DC 8
#define left_eye_CS 9
#define right_eye_DC 6
#define right_eye_CS 7

Adafruit_GC9A01A left_eye (left_eye_CS, left_eye_DC);
Adafruit_GC9A01A right_eye (right_eye_CS, right_eye_DC);
Servo left_ear, right_ear, left_eyelid, right_eyelid, beak;

#define BLACK      0x0000                                                               // some extra colors
#define BLUE       0x001F
#define RED        0xF800
#define GREEN      0x07E0
#define CYAN       0x07FF
#define MAGENTA    0xF81F
#define YELLOW     0xFFE0
#define WHITE      0xFFFF
#define ORANGE     0xFBE0
#define GREY       0x84B5
#define BORDEAUX   0xA000
#define AFRICA     0xAB21

void active_eyes(){
  static int radius = 24;
  static bool extension = true;
  static uint32_t time_start = millis();
  
  if (extension){
    passive_eyes(50);
    time_tick = 105;
  } else{
    passive_eyes(25);
    time_tick = 105;
  }
  if (radius_now == 25){
    extension = true;
  } else if (radius_now == 50){
    extension = false;
  }
}
void passive_eyes(int radius){
  static bool set = true;
  static uint32_t time_start = millis();
  
  if (radius_now < radius){
    if (millis() - time_start > time_tick){
      time_start = millis();
      left_eye.drawCircle(120, 120, radius_now, BLACK);
      right_eye.drawCircle(120, 120, radius_now, BLACK);
      radius_now ++;
    }
  } else if (radius_now > radius){
    if (millis() - time_start > time_tick){
      time_start = millis();
      left_eye.drawCircle(120, 120, radius_now, ORANGE);
      right_eye.drawCircle(120, 120, radius_now, ORANGE);
      radius_now --;
    }
  } else{
    left_eye.fillCircle(120, 120, radius_now, BLACK);
    right_eye.fillCircle(120, 120, radius_now, BLACK);
  }
  time_tick = 105;
}
void tallking(){
  left_ear_now = 34;
  right_ear_now = 30;
  left_eye_now = 85;
  right_eye_now = 80;
  beak_now = 98;
  passive_eyes(40);
  left_eyelid.write(left_eye_now);
  right_eyelid.write(right_eye_now);
  left_ear.write(left_ear_now);
  right_ear.write(right_ear_now);

  if (millis() % 900 < 500){
    beak.write(82);
  } else {
    beak.write(97);
  }
}
void uart(){
  if (Serial.available()){
    condition = int(Serial.read());
  }
  Serial.println(condition);
}
void servo_write(bool move_beak=1){
  left_eyelid.write(left_eye_now);
  right_eyelid.write(right_eye_now);
  left_ear.write(left_ear_now);
  right_ear.write(right_ear_now);
  beak.write(beak_now);
}
void surprise(){
  left_ear_now = 22;
  right_ear_now = 47;
  left_eye_now = 75;
  right_eye_now = 60;
  beak_now = 88;
  passive_eyes(45);
  blinking();
  servo_write();
}
void happinest(){
  left_ear_now = 34;
  right_ear_now = 38;
  left_eye_now = 85;
  right_eye_now = 80;
  beak_now = 93;
  passive_eyes(35);
  blinking();
  servo_write();
}
void sad(){
  left_ear_now = 50;
  right_ear_now = 24;
  left_eye_now = 130;
  right_eye_now = 128;
  beak_now = 98;
  passive_eyes(50);
  blinking();
  servo_write();
}
void blinking(){
  if (blink_update){
    time_blinking = random(1, 6) * 1000;
    wink = random(1, 13);
    update_time = millis();
    blink_update = false;
  } if (millis() - update_time > time_blinking + close_eyes){
    blink_update = true;
  }
  if (millis() % (time_blinking + close_eyes) > time_blinking){
    if (wink == 1){
      left_eye_now = 155;
      close_eyes = 300;
    } else if (wink == 2){
      right_eye_now = 155;
      close_eyes = 300;
    } else{
      left_eye_now = 155;
      right_eye_now = 155;
      close_eyes = 150;
    }
  }
  if (millis() % 16000 > 15200){
    left_ear_now = 22;
    right_ear_now = 47;
  }
}
void passive(){
  left_ear_now = 54;
  right_ear_now = 17;
  left_eye_now = 110;
  right_eye_now = 95;
  beak_now = 98;
  active_eyes();
  blinking();
  servo_write();
}
void loading(){
  if (condition == 4){
    if (loading_update){
      loading_update = false;
      //обновление дисплеев + моргание
      left_eye.setRotation (0); 
      right_eye.setRotation (0);
      left_eye_now = 170;
      right_eye_now = 155;
      servo_write();
      delay(50);
      radius_now = 20;
      left_eye.fillCircle(120, 120, 120, ORANGE);
      left_eye.fillCircle(120, 120, radius_now, BLACK);
      right_eye.fillCircle(120, 120, 120, ORANGE);
      right_eye.fillCircle(120, 120, radius_now, BLACK);
      delay(50);
      
      //удивление
      left_ear_now = 22;
      right_ear_now = 47;
      left_eye_now = 75;
      right_eye_now = 60;
      beak_now = 88;
      servo_write();
    }
    
    
    //сама анимация
    if (millis() - timer_waiting > 1){
      if (first_part){
        x += 0.00225 * 2;
        y = sqrt(1 - (x * x));
        left_eye.drawLine(120, 120, 120 - (int)(x * r), 120 - (int)(y * r), BLACK);
        right_eye.drawLine(120, 120, 120 - (int)(x * r), 120 - (int)(y * r), BLACK);
      } else{
        x -= 0.00205 * 2;
        y = sqrt(1 - (x * x));
        left_eye.drawLine(120, 120, 120 - (int)(y * r), 120 - (int)(x * r), BLACK);
        right_eye.drawLine(120, 120, 120 - (int)(y * r), 120 - (int)(x * r), BLACK);
      }
      if (x >= (sqrt(2) / 2) && first_part){
        first_part = false;
        x = sqrt(2) / 2;
      } else if (x <= 0 && !first_part){
        first_part = true;
        x = 0;
        y = sqrt(1 - (x * x));
        angle -= 45;
        left_eye.setRotation(angle);
        right_eye.setRotation(angle);
      }
      timer_waiting = micros();
    }
    if (angle == 0){
      loading_update = true;
      condition = 2;
      left_eye.fillCircle(120, 120, r - 1, BLACK);
      right_eye.fillCircle(120, 120, r - 1, BLACK);
      angle = 180;
    }
  }
}
void setup() {
  left_eyelid.attach(14);
  right_eyelid.attach(15);
  
  left_ear.attach(17);
  left_ear.write(64);
  right_ear.attach(16);
  right_ear.write(12);

  beak.attach(18);
  beak.write(beak_now);
  
  left_eye.begin();    
  right_eye.begin();
  SPI.begin();
  Serial.begin(9600); 
  
  left_eye.setRotation (0); 
  right_eye.setRotation (0);    
  left_eye.fillCircle(120, 120, 120, ORANGE);
  left_eye.fillCircle(120, 120, radius_now, BLACK);
  right_eye.fillCircle(120, 120, 120, ORANGE);
  right_eye.fillCircle(120, 120, radius_now, BLACK);
  //passive();
  delay(2000);
}
void loop (){
  uart();
  if (condition == 5){
    tallking();
    time_search = millis();
  } else if (condition == 6){
    if (millis() - time_search > 30000){
      sad();
    } else{
      passive();
    }
  } else if (condition == 4){
    loading();
    time_search = millis();
  } else{
    happinest();
  }

  
  //passive();
  
  /*uint32_t emotions_time = millis();
  while (millis() - emotions_time < 4500){
    uart();
    if ((millis() - emotions_time) % 5000 < 1000){surprise();}
    else if ((millis() - emotions_time) % 5000 <2000){happinest();}
    else if ((millis() - emotions_time) % 5000 < 3000){sad();}
    else{passive();}
  }*/
  //surprise();
}
