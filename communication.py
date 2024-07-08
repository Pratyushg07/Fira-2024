import time
import sys
import serial

def send_data():
    global TxData, TxD_packet
    lowbyte = TxData & 0xff
    TxD_packet[2] = lowbyte & 0xff
    TxD_packet[3] = ~lowbyte & 0xff
    highbyte = (TxData >> 8) & 0xff
    TxD_packet[4] = highbyte & 0xff
    TxD_packet[5] = ~highbyte & 0xff
    if (ser.write(TxD_packet) != 6):
        print("Transmit Error")
    time.sleep(0.1)

# Variables initialization
action = 0  # 0: Stop, 2: Left, 3: Right

# PYSERIAL section
highbyte = 0
lowbyte = 0
TxD_packet = bytearray(6)
TxD_packet[0] = 0xff
TxD_packet[1] = 0x55
TxD_packet[2] = lowbyte
TxD_packet[3] = ~lowbyte & 0xff  # to make sure ~lowbyte is contained within 1 byte
TxD_packet[4] = highbyte
TxD_packet[5] = ~highbyte & 0xff  # to make sure that ~highbyte is contained within 1 byte

ser = serial.Serial("/dev/ttyUSB0", 57600, timeout=0.1, write_timeout=0.1)  # via usb cable
ser.reset_output_buffer()  # clear buffer used for sending data to cm530

# MAIN LOOP STARTS HERE
while True:
    # User keyboard interface section
    print("Starting left operation for 5 seconds")
    action = 2  # Set to 2 for left
    TxData = 0

    if action == 2:
        TxData += 4  # adding button left to TxData
        end_time = time.time() + 5
        while time.time() < end_time:
            send_data()
            print("Left sent", TxData)

    print("Starting right operation for 5 seconds")
    action = 3  # Set to 3 for right
    TxData = 0

    if action == 3:
        TxData += 8  # adding button right to TxData
        end_time = time.time() + 5
        while time.time() < end_time:
            send_data()
            print("Right sent", TxData)

    print("Stopping operation")
    action = 0  # Set to 0 to stop
    TxData = 0
    send_data()
    break

sys.exit()
