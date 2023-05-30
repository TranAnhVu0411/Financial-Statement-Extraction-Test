import cv2
import numpy as np
import json
from text_line_detection_utils import *
from scipy.signal import argrelmin

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import os
from tqdm import tqdm

def text_region_detection(img_array):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,4))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    metadata = []
    for c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c[1])
        metadata.append({'text-region': (x, y, w, h)})
    return metadata

def text_line_detection(img_array):
    result = img_array.copy()
    # Convert color image to gray image
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Rotate image
    img_array = np.transpose(img_array)
    # Filtering image
    imgFiltered1 = cv2.filter2D(img_array, -1, createKernel(), borderType=cv2.BORDER_REPLICATE)
    # Normalize image
    img_array = normalize(imgFiltered1)
    # Get mean and standard deviation
    summ = applySummFunctin(img_array)
    smoothed = smooth(summ, 35)
    mins = argrelmin(smoothed, order=2)
    arr_mins = np.array(mins)
    coordinate = get_line_coordinate(result, arr_mins[0])
    return coordinate

def ocr(img_array, detector):
    img_pil = Image.fromarray(img_array)
    s = detector.predict(img_pil)
    return s

if __name__=='__main__':
    # Load vietOCR model
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)

    img_idx = 7
    img = cv2.imread('result/preprocess/test{}-preprocess.jpg'.format(img_idx))
    # Create directory
    if not os.path.exists('region/test{}'.format(img_idx)):
        os.makedirs('region/test{}'.format(img_idx))
    if not os.path.exists('lines/test{}'.format(img_idx)):
        os.makedirs('lines/test{}'.format(img_idx))

    print('GET TEXT REGION')
    text_metadata = text_region_detection(img)
    new_text_metadata = []
    print('GET LINE REGION AND OCR')
    for idx, metadata in tqdm(enumerate(text_metadata)):
        x_text = metadata['text-region'][0]
        y_text = metadata['text-region'][1]
        w_text = metadata['text-region'][2]
        h_text = metadata['text-region'][3]
        text_region = img[y_text:y_text+h_text, x_text:x_text+w_text]
        cv2.imwrite('region/test{}/{}.jpg'.format(img_idx, idx), text_region)
        lines = text_line_detection(text_region)
        lines_metadata = []
        for idx2, line in enumerate(lines):
            x_line = line[0]
            y_line = line[1]
            w_line = line[2]
            h_line = line[3]
            line_region = text_region[y_line:y_line+h_line, x_line:x_line+w_line]
            cv2.imwrite('lines/test{}/region_{}_line_{}.jpg'.format(img_idx, idx, idx2), line_region)
            text = ocr(line_region, detector)
            lines_metadata.append({'line_coordinates': [int(x_line), int(y_line), int(w_line), int(h_line)], 'text': text})
        new_text_metadata.append({'text_coordinates': [int(x_text), int(y_text), int(w_text), int(h_text)], 'lines': lines_metadata})

    with open('result/ocr/test{}-metadata.json'.format(img_idx), 'w') as f:
        json.dump(new_text_metadata, f)


