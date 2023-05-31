from CRAFT_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg
from CRAFT_pytorch.craft_utils import getDetBoxes, adjustResultCoordinates
from torch.autograd import Variable
import torch
import numpy as np
import cv2

from text_line_detection_utils import *
from scipy.signal import argrelmin

from PIL import Image

def test_net(net, image, text_threshold, link_threshold, low_text, poly, cuda=False, refine_net=None, canvas_size=1280, mag_ratio=1.5):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def get_text_object(img, net):
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    text_threshold=0.7
    link_threshold=0.4
    low_text = 0.4
    poly = False
    refine_net = None

    boxes, _, _ = test_net(net, img, text_threshold, link_threshold, low_text, poly, refine_net)
    tokens_in_table = [np.array([box[0].astype('int32'), box[2].astype('int32')]).flatten().tolist() for box in boxes]
    return tokens_in_table

def text_region_detection(img_array):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    metadata = []
    for c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c[1])
        metadata.append({'text-region': (x, y, w, h)})
    return metadata

def text_line_detection(img_array, min_line_height):
    img_copy = img_array.copy()
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
    coordinate = get_line_coordinate(img_copy, arr_mins[0])
    # post process line (if there is only one lines, use whole image, otherwise, use postprocess coordinate)
    new_coordinate = []
    for coord in coordinate:
        y = coord[1]
        h = coord[3]
        if h>min_line_height/2 and y<img_copy.shape[0]:
            new_coordinate.append(coord)
    if len(new_coordinate)==1:
        w = img_copy.shape[1]
        h = img_copy.shape[0]
        return [[0, 0, w, h]]
    else:
        return new_coordinate

# remove white space in left and right around line
def preprocess_line_region(line):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    min_x_list = []
    width = 0
    for c in enumerate(cnts):
        x,_,w,_ = cv2.boundingRect(c[1])
        min_x_list.append(x)
        width += w
    min_x = min(min_x_list)
    return min_x, width

import time
def ocr_cell(img, min_line_height, detector):
    result = ''
    # cv2.imwrite('cells/{}.jpg'.format(time.time()*1000), img)
    text_metadata = text_region_detection(img)
    for metadata in text_metadata:
        x_text = metadata['text-region'][0]
        y_text = metadata['text-region'][1]
        w_text = metadata['text-region'][2]
        h_text = metadata['text-region'][3]
        text_region = img[y_text:y_text+h_text, x_text:x_text+w_text]
        # cv2.imwrite('text-region/{}.jpg'.format(time.time()*1000), text_region)
        lines = text_line_detection(text_region, min_line_height)

        for line in lines:
            x_line = line[0]
            y_line = line[1]
            w_line = line[2]
            h_line = line[3]
            line_region = text_region[y_line:y_line+h_line, x_line:x_line+w_line]
            # cv2.imwrite('lined-region/{}.jpg'.format(time.time()*1000), line_region)
            new_x_line, new_w_line = preprocess_line_region(line_region)
            new_line_region = text_region[y_line:y_line+h_line, new_x_line:new_x_line+new_w_line]
            # cv2.imwrite('lines-region/{}.jpg'.format(time.time()*1000), new_line_region)
            img_pil = Image.fromarray(new_line_region)
            text = detector.predict(img_pil)
            result+=text+' '
    return result
