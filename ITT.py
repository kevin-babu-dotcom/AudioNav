from PIL import Image
import pytesseract
import cv2
from GTTS import gtts


def reader():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    capr = cv2.VideoCapture(os.environ.get('CAM_IP1'))
    retr, framer = capr.read()
    framer = cv2.rotate(framer, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(r"C:\Users\aaron\OneDrive\Desktop\text.jpg", framer)
    capr.release()

    confidence_threshold = 80

    image_path = r"C:\Users\aaron\OneDrive\Desktop\text.jpg" 
    image = Image.open(image_path)
    result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    print(result)
    filtered_text = [result['text'][i] for i, conf in enumerate(result['conf']) if conf >= confidence_threshold]

    print("Extracted Text:")
    text=''
    for i in filtered_text:
        text+=i+" "

    gtts(text, r'E:\Pegasus\imageaudio.mp3')
