import serial

port = '/dev/ttyUSB0'
baud_rate = 9600
timeout = 1
ser = serial.Serial(port, baud_rate, timeout=timeout)

ser.write(input().encode("ascii"))