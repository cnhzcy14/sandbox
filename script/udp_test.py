import socket

UDP_IP = "192.168.2.31"
UDP_PORT = 8000

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(26) # buffer size is 1024 bytes
    print(addr)
    print("received message: %02x" % data[3])
    print("received message: %02x" % data[25])
    
    # i = int(data[3].__str__, 16)
    # string = str(data)
    # print(string)
    # databyte = data[4:5]
    # print("camera %d" % (int(data[4])>>6))

    # print("camera %d" % (int(data[11])))
    # int(bin(data)[4:12], 64)
    # databyte = int(data[4:6].__str__, 16)
    # string = str(databyte)
    # print("received message: %s\n" % databyte)

