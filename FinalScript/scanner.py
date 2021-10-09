import numpy as nm
import pytesseract
import cv2
import mss
import keyboard
import time
import re
from collections import Counter
import joblib

def main():
    with mss.mss() as sct:
        # path of image to text scanner
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # 1080p default support
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        # load the pretrained file through joblib
        trained = joblib.load('finalmodel.sav')
        # supported emotions
        emoji_dict = {"joy": "😂", "fear": "😱", "anger": "😠",
                      "sadness": "😢", "disgust": "😒", "shame": "😳", "guilt": "😳"}
        # loop begins
        while(True):
            # prevents the output of / key due to its function as a toggle button
            # keyboard.block_key('/')
            # toggle says go
            if keyboard.is_pressed('ctrl+enter'):
                # benchmark time after keypress
                start_time = time.time()
                # default discord text area
                final = {"top": 975, "left": 375, "width": 840, "height": 40}
                # one screenshot to find the program
                fullimg = pytesseract.image_to_string(cv2.cvtColor(
                    nm.array(sct.grab(monitor)), cv2.COLOR_BGR2GRAY), lang='eng')
                # facebook check
                if 'Facebook' in fullimg:
                    # fb text area
                    final = {"top": 1000, "left": 448,
                             "width": 992, "height": 23}
                # whatsapp check
                elif 'WhatsApp' in fullimg:
                    # whatsapp text area
                    final = {"top": 995, "left": 693,
                             "width": 1142, "height": 30}
                # analyzes the typed text visually using tesseract
                txtstr = pytesseract.image_to_string(cv2.cvtColor(
                    nm.array(sct.grab(final)), cv2.COLOR_BGR2GRAY), lang='eng')
                # algorithm for creating emoji prediction
                features = create_feature(txtstr, nrange=(1, 4))
                features = trained[0].transform(features)
                prediction = trained[1].predict(features)[0]
                # prints text, emoji, time taken
                print(txtstr, emoji_dict[prediction], 'The loop took: {0}'.format(
                    time.time()-start_time))
                # put the emoji into your chat along with a spacer character.
                keyboard.press('space')
                if prediction == 'joy':
                    keyboard.write('😂')
                elif prediction == 'fear':
                    keyboard.write('😱')
                elif prediction == 'anger':
                    keyboard.write('😠')
                elif prediction == 'sadness':
                    keyboard.write('😢')
                elif prediction == 'disgust':
                    keyboard.write('😒')
                elif prediction == 'shame':
                    keyboard.write('😳')
                elif prediction == 'guilt':
                    keyboard.write('😳')


# helpermethod for tokenization
def ngram(token, n):
    output = []
    for i in range(n-1, len(token)):
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram)
    return output


# helper method for sentiment analysis
def create_feature(text, nrange=(1, 1)):
    text_features = []
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)


# calling function
main()
