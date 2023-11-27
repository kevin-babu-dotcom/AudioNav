import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from Distance import detect
import cv2
import keyboard
from GTTS import gtts
from ITT import reader

load_dotenv()


while True:
    if keyboard.is_pressed('p'):
        print(1)
        try:
            capl = cv2.VideoCapture(os.environ.get('CAM_IP1'))
            retl, framel = capl.read()
            framel = cv2.rotate(framel, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(r"C:\Users\aaron\OneDrive\Desktop\capturer.jpg", framel)
            capl.release()
        except:
            print("right camera image error")


        try:
            capr = cv2.VideoCapture(os.environ.get('CAM_IP2'))
            retr, framer = capr.read()
            framer = cv2.rotate(framer, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(r"C:\Users\aaron\OneDrive\Desktop\capturel.jpg", framer)
            capr.release()
        except:
            print("left camer image error")



        dist = (detect(r"C:\Users\aaron\OneDrive\Desktop\capturel.jpg", r"C:\Users\aaron\OneDrive\Desktop\capturer.jpg"))
        steps = str(int(dist//50)+2)
        talk = "a person is "+steps+" steps in front of you"
        print(talk)
        gtts(talk,r'E:\Pegasus\distaudio.mp3')
    elif keyboard.is_pressed('i'):
        print(2)
        try:
            reader()
        except:
            pass


    elif keyboard.is_pressed('q'):
        break