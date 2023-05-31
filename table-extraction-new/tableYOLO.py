import postprocess
from preprocess import *
import xml.etree.ElementTree as ET
from OCR import ocr_cell, get_text_object
import cv2
import torch
from html2excel import ExcelParser

import sys
sys.path.insert(0, 'CRAFT_pytorch')
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.test import copyStateDict

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from tqdm import tqdm
from pathlib import Path

structure_class_names = [
    'table column', 'table column-header', 'table projected row header', 'table row', 'table spanning cell'
]
structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
structure_class_thresholds = {
    "table column": 0,
    "table column header": 0,
    "table projected row header": 0,
    "table row": 0,
    "table spanning cell": 0,
}

def table_structure_recognition(image, model):
    print('TABLE STRUCTURE DETECTION')
    pred = model(image, size=640)
    pred = pred.xywhn[0]    
    result = pred.numpy()
    return result

def convert_structure(img, str_result, tokens_in_table):
    print('CLEAN TABLE STRUCTURE')
    width = img.shape[1]
    height = img.shape[0]
    bboxes = []
    scores = []
    labels = []
    for item in str_result:
        class_id = int(item[5])
        score = float(item[4])
        min_x = item[0]
        min_y = item[1]
        w = item[2]
        h = item[3]
        
        x1 = int((min_x-w/2)*width)
        y1 = int((min_y-h/2)*height)
        x2 = int((min_x+w/2)*width)
        y2 = int((min_y+h/2)*height)

        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(class_id)

    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({'bbox': bbox, 'score': score, 'label': label})
        
    table = {'objects': table_objects, 'page_num': 0}
    table_structures, cells, confidence_score = postprocess.objects_to_cells(table, table_objects, tokens_in_table, structure_class_names, structure_class_thresholds)
    return table_structures, cells, confidence_score

def cells_to_html(img, cells, ocr_model):
    print('OCR')
    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element("table")
    current_row = -1

    for idx, cell in tqdm(enumerate(cells)):
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            cell_tag = "td"
            row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        text = ''
        if len(cell['spans'])!=0:
            # Calculate min line height by using text height
            min_line_height = min([bb[3]-bb[1] for bb in cell['spans']])/2
            bbox = cell['bbox']
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            crop = img[y1:y2, x1:x2]
            # cv2.imwrite('cell{}.jpg'.format(idx), crop)
            text = ocr_cell(crop, min_line_height, ocr_model)

        tcell.text = text

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))

def visualize(image, cells):
    for i, cell in enumerate(cells):
        bbox = cell['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        col_num = cell['column_nums'][0]
        row_num = cell['row_nums'][0]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0,255,0))
        cv2.putText(image, str(row_num)+'-'+str(col_num), (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255))
    cv2.imwrite("result/structure/{}.jpg".format(Path(image_path).stem), image)

if __name__=='__main__':
    image_path = 'image/01a93bdc-CT_CP_Van_Hoa_Phuong_Nam-page-14-table00.jpg'
    image = cv2.imread(image_path)
    image = deskew(image)
    deskew_image = image.copy()
    cv2.imwrite("result/preprocess/deskew/preprocess_{}.jpg".format(Path(image_path).stem), image)
    str_model = torch.hub.load(
        'yolov5',
        'custom',
        source='local',
        path = 'model/best.pt',
        force_reload = True
    )

    str_result = table_structure_recognition(image, str_model)

    net = CRAFT()
    craft_model = 'model/craft_mlt_25k.pth'
    net.load_state_dict(copyStateDict(torch.load(craft_model, map_location='cpu')))
    net.eval()
    tokens_in_table = get_text_object(image, net)
    _, cells, _ = convert_structure(image, str_result, tokens_in_table)
    visualize(deskew_image, cells)

    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)
    image = remove_line(image)
    cv2.imwrite("result/preprocess/remove_line/preprocess_{}.jpg".format(Path(image_path).stem), image)
    html_text = cells_to_html(image, cells, detector)

    with open("result/html/{}.html".format(Path(image_path).stem), "w") as f:
        f.write(html_text)
        f.close()
    parser = ExcelParser("result/html/{}.html".format(Path(image_path).stem))
    parser.to_excel("result/excel/{}.xlsx".format(Path(image_path).stem))