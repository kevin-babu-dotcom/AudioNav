
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from GTTS import gtts
import time
from dotenv import load_dotenv


load_dotenv()



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
modelpath=r"E:\Pegasus\face detection\keras_model.h5"
model = load_model(modelpath,compile=False)

# Load the labels
cls_name_path=r'E:\Pegasus\face detection\labels.txt'
class_names = open(cls_name_path, "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(os.environ.get('CAM_IP1'))


name = ""
while True:
    
    # Grab the webcamera's image.
    ret, frame = camera.read()
    image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Check if the image was captured successfully
    if not ret:
        print("Failed to capture image from the camera.")
        break


    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Convert the predicted class to speech
    if 'Background' not in class_name and name!=class_name and 'Strangers' not in class_name:
        name = class_name
        gtts("The predicted class is " + class_name[2:],r'E:\Pegasus\Facaudio.mp3')

        

    # Listen to the keyboard for presses.

    # 27 is the ASCII for the esc key on your keyboard.


camera.release()
cv2.destroyAllWindows()
