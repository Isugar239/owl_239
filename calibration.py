import serial

port = '/dev/ttyUSB0'
baud_rate = 9600
timeout = 1
ser = serial.Serial(port, baud_rate, timeout=timeout)

input()
ser.write("7".encode("ascii"))
input()
ser.write("7".encode("ascii"))
