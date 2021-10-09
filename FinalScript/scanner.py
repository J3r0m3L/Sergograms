import joblib
import numpy as nm
import pytesseract
import cv2
import mss
import keyboard
import time
import re
from collections import Counter

def main():
    with mss.mss() as sct:
        # path of exec
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        trained = joblib.load('finalmodel.sav')
        emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}
        while(True):
            keyboard.block_key('/')
            # toggle key/check for which program
            if keyboard.is_pressed('/'):
                start_time = time.time()
                final = {"top": 975, "left": 375, "width": 840, "height": 40}
                fullimg = pytesseract.image_to_string(cv2.cvtColor(
                    nm.array(sct.grab(monitor)), cv2.COLOR_BGR2GRAY), lang='eng')
                if 'Facebook' in fullimg:
                    #fbtxt = ImageGrab.grab(bbox=(448, 1000, 1440, 1023))
                    final = {"top": 1000, "left": 448,
                             "width": 992, "height": 23}
                elif 'WhatsApp' in fullimg:
                    #whattxt = ImageGrab.grab(bbox=(693, 995, 1835, 1025))
                    final = {"top": 995, "left": 693,
                             "width": 1142, "height": 30}
                txtstr = pytesseract.image_to_string(cv2.cvtColor(
                    nm.array(sct.grab(final)), cv2.COLOR_BGR2GRAY), lang='eng')
                features = create_feature(txtstr, nrange=(1, 4))
                features = trained[0].transform(features)
                prediction = trained[1].predict(features)[0]
                print(txtstr,emoji_dict[prediction],'The loop took: {0}'.format(time.time()-start_time))

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

main()
