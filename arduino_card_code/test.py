import serial

# Replace 'COM3' with your actual COM port
# On Linux/Mac, it might be something like '/dev/ttyUSB0' or '/dev/ttyS0'
serial_port = 'COM3'
baud_rate = 115200  # Ensure this matches the baud rate set in the ESP32 code

ser = serial.Serial(serial_port, baud_rate, timeout=1)

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        if line:
            print(line)
            # You can parse the data further here
            data_parts = line.split(',')
            if len(data_parts) == 2:
                distance_str, led_str = data_parts
                distance_value = distance_str.split(':')[1]
                # led_status = led_str.split(':')[1]
                print(f"Distance: {distance_value} cm")

ser.close()
