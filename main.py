from keras.models import load_model
import numpy as np
import cv2
import pyautogui
import time

model = load_model('Model/Keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cap = cv2.VideoCapture(0)

classes = ['Pausar','Up','Down','Right','Left','Nan','F','altf4']
    
def pause_video():
    pyautogui.hotkey('space')
    time.sleep(1)  
    

def down():
    pyautogui.hotkey('down')


def up():
    pyautogui.hotkey('up')


def left():
    pyautogui.hotkey('left')
    time.sleep(0.5)

def right():
    pyautogui.hotkey('Right')
    time.sleep(0.5)

def f():
    pyautogui.hotkey('F')
    time.sleep(0.5)    

def altf4():
    pyautogui.hotkey('alt','f4')
    time.sleep(2)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img, (224, 224))
    image_array = np.asarray(imgS)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    indexVal = np.argmax(prediction)

    cv2.putText(img, str(classes[indexVal]),(50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    time.sleep(0.2)
    print(classes[indexVal])  
    
    cv2.imshow('img',img)
    cv2.waitKey(1)

    match classes[indexVal]:
        case "Pausar":
            pause_video()

        case "Down":
            down()

        case "Up" :
           up()

        case "Left":
           left()

        case "Right":
            right()

        case "F":
            f()

        case "altf4":
            altf4()             

        case _:
            print('Nan')       
        
