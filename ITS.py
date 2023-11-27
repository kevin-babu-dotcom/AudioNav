import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2, easyocr
import matplotlib.pyplot as plt 
from GTTS import gtts


def ITS():
    try:
        capl = cv2.VideoCapture(os.environ.get('CAM_IP1'))
        retl, framel = capl.read()
        framel = cv2.rotate(framel, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(r"C:\Users\aaron\OneDrive\Desktop\text.jpg", framel)
        capl.release()
    except:
        print("camera image error")


    image_path = r"C:\Users\aaron\OneDrive\Desktop\text.jpg"
    img = cv2.imread(image_path)


    reader = easyocr.Reader(['en'] , gpu=False)

    list=[]
    text = reader.readtext(img)
    finaltext=''
    for t in text:
        if t[2] > 0.9:
            finaltext+=t[1]
    print(finaltext)
    gtts(finaltext,r'E:\Pegasus\imageaudio.mp3')