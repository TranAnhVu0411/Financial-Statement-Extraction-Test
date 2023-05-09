import cv2
import pytesseract
import numpy as np
from wand.image import Image as wand_image
import camelot
import json
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import pandas as pd
import os

import sys
# Thay đổi đường dẫn đến CRAFT_pytorch
sys.path.insert(0, 'CRAFT_pytorch')
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg
from CRAFT_pytorch.craft_utils import getDetBoxes, adjustResultCoordinates
from CRAFT_pytorch.test import copyStateDict
from collections import OrderedDict
import torch
from torch.autograd import Variable

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

def get_text_bb(img, net):
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    text_threshold=0.7
    link_threshold=0.4
    low_text = 0.4
    poly = False
    refine_net = None

    boxes, polys, score_text = test_net(net, img, text_threshold, link_threshold, low_text, poly, refine_net)
    return boxes

def preprocess_image(image_path, grid=False):
    # Deskew image
    with wand_image(filename=image_path) as img:
        img.deskew(0.4*img.quantum_range)
        rotated = np.array(img)

    result = rotated.copy()

    # Remove line in no grid table
    if not grid :
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255), 5)

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # cv2.imwrite('image/preprocess/test{}.png'.format(100), result)
    
    return result

def ocr_table_structure(image, pdf_path):
    # ocr and save as pdf
    pdf = pytesseract.image_to_pdf_or_hocr(image, lang='vie', extension='pdf', config='--psm 4')
    with open(pdf_path, 'w+b') as f:
        f.write(pdf) # pdf type is bytes by default

def get_table_structure(pdf_path, grid=False):
    # Get table structure
    if not grid:
        tables = camelot.read_pdf(pdf_path, flavor='stream', edge_tol=1000, row_tol=20, strip_text='.\n')
    else:
        tables = camelot.read_pdf(pdf_path, flavor='lattice')

    # Save table metadatas
    cell_metadata = []
    for idx, table in enumerate(tables):
        cell_metadata.append([])
        for row in table.cells:
            for cell in row:
                cell_metadata[idx].append({'x1': cell.x1, 'y1': cell.y1, 'x2': cell.x2, 'y2': cell.y2})
    # with open('metadata/table/metadata{}.json'.format(100), 'w') as f:
    #     json.dump(cell_metadata, f)
    return cell_metadata

def convert_pdf_meta_to_image_meta(pdf_metadata, image):
    # https://github.com/atlanhq/camelot/issues/172
    image_metadata=[]
    for idx, table in enumerate(pdf_metadata):
        image_metadata.append([])
        for cell in table:
            x1=abs(int(cell['x1']*(0.9723)))
            y1=abs(image.shape[0]-int(cell['y2']*(0.9723)))
            x2=abs(int(cell['x2']*(0.9723)))
            y2=abs(image.shape[0]-int(cell['y1']*(0.9723)))
            image_metadata[idx].append([x1, y1, abs(x1-x2), abs(y1-y2)])
    return image_metadata

def convert_table_structure_to_csv(image, table_box, ocr_detector, text_detector, csv_path, name):
    # REFERENCES: https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec
    #Creating a list of heights for all detected boxes
    heights = [table_box[i][3] for i in range(len(table_box))]

    #Get mean of heights
    mean = np.mean(heights)

    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(table_box)):    
            
        if(i==0):
            column.append(table_box[i])
            previous=table_box[i]    
        
        else:
            if(table_box[i][1]<=previous[1]+mean/2):
                column.append(table_box[i])
                previous=table_box[i]            
                
                if(i==len(table_box)-1):
                    row.append(column)        
                
            else:
                row.append(column)
                column=[]
                previous = table_box[i]
                column.append(table_box[i])
    
    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

    center=np.array(center)
    center.sort()
    #Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    #Merge cell that contains more than one bounding box
    mergeboxes = []
    for i in range(len(finalboxes)):
        mergeboxes.append([])
        for j in range(len(finalboxes[i])):
            mergeboxes[i].append([])
            if len(finalboxes[i][j])>1:
                x = min([tempbox[0] for tempbox in finalboxes[i][j]])
                y = min([tempbox[1] for tempbox in finalboxes[i][j]])
                w = max([tempbox[2] for tempbox in finalboxes[i][j]])
                h = sum([tempbox[3] for tempbox in finalboxes[i][j]])
                mergeboxes[i][j].append([x, y, w, h])
            else:
                mergeboxes[i][j].append(finalboxes[i][j][0])
    
    # with open('metadata/mergebox.json', 'w') as f:
    #     json.dump(mergeboxes, f)
    
    #from every single image-based cell/box the strings are extracted via vietocr and stored in a list
    outer=[]
    for i in range(len(mergeboxes)):
        for j in range(len(mergeboxes[i])):
            inner=''
            if(len(mergeboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(mergeboxes[i][j])):
                    y,x,w,h = mergeboxes[i][j][k][0],mergeboxes[i][j][k][1], mergeboxes[i][j][k][2],mergeboxes[i][j][k][3]
                    finalimg = image[x:x+h, y:y+w]
                    # cv2.imwrite('image/word/word-{}-{}.png'.format(i, j), finalimg)
                    bbox = get_text_bb(finalimg, text_detector)
                    out = "none"
                    if len(bbox)!=0:
                        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        # border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                        # resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        # dilation = cv2.dilate(resizing, kernel,iterations=1)
                        # erosion = cv2.erode(dilation, kernel,iterations=2)
                        # convert_color = cv2.cvtColor(finalimg, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(finalimg)
                        out =  ocr_detector.predict(img_pil)
                        print(i, j, out)

                    inner = inner +" "+ out
                outer.append(inner)
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    dataframe.to_csv(os.path.join(csv_path, '{}.csv'.format(name)), index=False)

# if '__name__' == '__main__':
img_idx = 1
image_path = 'image/table/test{}.jpg'.format(img_idx)
pdf_path = 'pdf/test{}.pdf'.format(img_idx)
csv_path = 'csv/pipeline'

grid=False
print('PREPROCESS IMAGE')
image = preprocess_image(image_path, grid)
print('OCR TABLE STRUCTURE')
ocr_table_structure(image, pdf_path)
print('GET TABLE STRUCTURE')
pdf_table_metadata = get_table_structure(pdf_path, grid)
print('CONVERT PDF METADATA TO IMAGE METADATA')
image_table_metadata = convert_pdf_meta_to_image_meta(pdf_table_metadata, image)

# Load vietOCR model
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)

# Load CRAFT model
net = CRAFT()
net.load_state_dict(copyStateDict(torch.load('/Users/trananhvu/Documents/GitHub/Financial-Statement-Extraction-Test/model/craft_mlt_25k.pth', map_location='cpu')))
net.eval()

print('OCR CELL AND EXTRACT TO CSV')
for idx, table_box in enumerate(image_table_metadata):
    convert_table_structure_to_csv(image, table_box, detector, net, csv_path, 'test{}-{}'.format(img_idx, idx))
