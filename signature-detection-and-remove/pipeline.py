import cv2
import torch
import numpy as np
import os

import sys
sys.path.insert(0, 'signver')

from signver.cleaner import Cleaner
from signver.utils.data_utils import resnet_preprocess

# Load detection model
print('SIGNATURE DETECTION')
detection_model = torch.hub.load(
    'yolov5',
    'custom',
    source='local',
    path = 'model/best.pt',
    force_reload = True)

img_path = '../preprocess-document/result-preprocess'
img_name = 'test6removestamp.jpg'
 
img = cv2.imread(os.path.join(img_path, img_name))
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV image (BGR to RGB)

results = detection_model([img], size=640)  # includes NMS

signature_metadata = []
if len(results.xyxy[0])!=0:    
    for sign_coord in results.xyxy[0]:
        x0 = int(sign_coord[0])
        y0 = int(sign_coord[1])
        x1 = int(sign_coord[2])
        y1 = int(sign_coord[3])
        signature_metadata.append({
            'coordinates': (x0, y0, x1, y1),
            'crop_image': cv2.cvtColor(img_copy[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        })

print('SIGNATURE CLEANING')
prethreshold_metadata = []
mask_metadata = []
if len(signature_metadata)!=0:
    # Load clean model
    cleaner_model_path = "signver/models/cleaner/small"
    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)

    signatures = []
    for metadata in signature_metadata:
        signatures.append(metadata['crop_image'])

    # Feature extraction with resnet model
    sigs= [ resnet_preprocess( x, resnet=False, invert_input=False ) for x in signatures ]

    # Normalization and clean
    norm_sigs = [ x * (1./255) for x in sigs]
    cleaned_sigs = cleaner.clean(np.array(norm_sigs))

    # Reverse normalization
    rev_norm_sigs = [ x / (1./255) for x in cleaned_sigs]

    # Resize and binarization
    for i in range(len(rev_norm_sigs)):
        img_resize = cv2.resize(
            rev_norm_sigs[i],
            (signatures[i].shape[1], signatures[i].shape[0]),
            interpolation = cv2.INTER_CUBIC
        )
        img_gray = cv2.cvtColor(img_resize.astype('uint8'), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        mask_metadata.append({
            'coordinates': signature_metadata[i]['coordinates'],
            'mask': mask
        })

print('SAVE RESULT')
if len(mask_metadata)!=0:
    img_mask = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
    for metadata in mask_metadata:
        x0 = metadata['coordinates'][0]
        y0 = metadata['coordinates'][1]
        x1 = metadata['coordinates'][2]
        y1 = metadata['coordinates'][3]
        img_mask[y0:y1, x0:x1] = metadata['mask']

    cv2.imwrite(os.path.join('result/mask', '{}-mask.jpg'.format(img_name.split('.')[0])), img_mask)
    img_copy = (img_copy*(np.expand_dims(cv2.bitwise_not(img_mask)/255, axis=2))+np.expand_dims(img_mask, axis=2)).astype('uint8')
    cv2.imwrite(os.path.join('result/preprocess', '{}-result.jpg'.format(img_name.split('.')[0])), img_copy)

    
