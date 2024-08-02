import cv2
import numpy as np
from pyzbar.pyzbar import decode
import serial

def sendToArduino(message):
    serialcmd = str(message) + "\n"
    ser.write(serialcmd.encode())                    
    # line = ser.readline().decode("utf-8").rstrip() 
    # print(line)                                    

def qrReader():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    with open("/home/beesh/Desktop/pyPro/dmt2f/myDatafile.txt") as f:
        myDatalist = f.read().splitlines()

    while True:
        print("working....")
        success, img = cap.read()
        for barcode in decode(img):
            myData = barcode.data.decode("utf-8")
            print(myData)
            if myData in myDatalist:
                myOutput = "Authorized"
                myColor = (0, 255, 0)
                return 1

            else:
                myOutput = "Un-Authorized"
                myColor = (0, 0, 255)

            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, myColor, 5)
            pts2 = barcode.rect
            cv2.putText(
                img,
                myOutput,
                (pts2[0], pts2[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                myColor,
                2,
            )

        cv2.imshow("result", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0", baudrate=57600, timeout=1)
    ser.flush
    if qrReader() == 1:
        sendToArduino("1")
    cv2.waitKey(1)