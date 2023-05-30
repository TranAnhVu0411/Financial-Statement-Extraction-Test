import cv2
import numpy as np
from wand.image import Image as wand_image
from PIL import Image, ImageEnhance

import torch
import os

import sys
sys.path.insert(0, 'signver')
from signver.cleaner import Cleaner
from signver.utils.data_utils import resnet_preprocess

sys.path.insert(1, 'CRAFT_pytorch')
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg
from CRAFT_pytorch.craft_utils import getDetBoxes, adjustResultCoordinates
from CRAFT_pytorch.test import copyStateDict
from torch.autograd import Variable

# Use cv2 image format
def deskew(img_array):
    img_str = cv2.imencode('.jpg', img_array)[1].tobytes()
    print('DESKEW')
    # Deskew
    with wand_image(blob=img_str) as img:
        img.deskew(0.4*img.quantum_range)
        deskew = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return deskew

def contrast_adjustment(img_array):
    print('INCREASE CONTRAST')
    # Increase contrast
    im = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(im)
    factor = 1.5 #increase contrast
    contrast = np.array(enhancer.enhance(factor))
    return contrast

def remove_stamp(img_array):
    print('REMOVE STAMP')
    # Remove stamp
    remove_stamp = img_array.copy()
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    
    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_array, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([155,25,0])
    upper_red = np.array([179,255,255])
    mask1 = cv2.inRange(img_array, lower_red, upper_red)

    # join masks
    mask = mask0+mask1

    remove_stamp = (remove_stamp*(np.expand_dims(cv2.bitwise_not(mask)/255, axis=2))+np.expand_dims(mask, axis=2)).astype('uint8')
    return remove_stamp

def signature_remove(img_array, detection, cleaner):
    # Load detection model
    print('SIGNATURE DETECTION')
    img_copy = img_array.copy()
    results = detection([img_array], size=640)  # includes NMS
    signature_metadata = []
    if len(results.xyxy[0])!=0:    
        for sign_coord in results.xyxy[0]:
            x0 = int(sign_coord[0])
            y0 = int(sign_coord[1])
            x1 = int(sign_coord[2])
            y1 = int(sign_coord[3])
            signature_metadata.append({
                'coordinates': (x0, y0, x1, y1),
                'crop_image': img_copy[y0:y1, x0:x1]
            })

    print('SIGNATURE CLEANING')
    mask_metadata = []
    if len(signature_metadata)!=0:
        # Load clean model

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
            ).astype('uint8')
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
            mask_metadata.append({
                'coordinates': signature_metadata[i]['coordinates'],
                'mask': mask
            })

    if len(mask_metadata)!=0:
        img_mask = np.zeros((img_copy.shape[0], img_copy.shape[1])).astype('uint8')
        for metadata in mask_metadata:
            x0 = metadata['coordinates'][0]
            y0 = metadata['coordinates'][1]
            x1 = metadata['coordinates'][2]
            y1 = metadata['coordinates'][3]
            img_mask[y0:y1, x0:x1] = metadata['mask']

        img_copy = (img_copy*(np.expand_dims(cv2.bitwise_not(img_mask)/255, axis=2))+np.expand_dims(img_mask, axis=2)).astype('uint8')
    return img_copy

def remove_background(img_array, net, dilate=True):
    print('REMOVING BACKGROUND')
    def test_net(net, image, text_threshold, link_threshold, low_text, poly, cuda=False, refine_net=None, canvas_size=1280, mag_ratio=1.5):
        # resize
        img_resized, target_ratio, _ = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
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
    
    # CRAFT parameters
    text_threshold=0.7
    link_threshold=0.4
    low_text = 0.4 #0.4
    poly = False
    refine_net = None
    # Text box detection
    bboxes, _, _ = test_net(net, img_array, text_threshold, link_threshold, low_text, poly, refine_net)
   
    # Create mask
    mask = np.zeros((img_array.shape[0], img_array.shape[1]))
    color = 255
    for box in bboxes:
        mask = cv2.rectangle(mask, box[0].astype('int32'), box[2].astype('int32'), color, -1)
    
    if dilate:
        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
        mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.bitwise_not(mask.astype('uint8')) # Inverse mask

    img_array = (img_array*(np.expand_dims(cv2.bitwise_not(mask)/255, axis=2))+np.expand_dims(mask, axis=2)).astype('uint8')
    return img_array


if __name__=='__main__':
    img_idx = 7
    img = cv2.imread('image/test{}.jpg'.format(img_idx))
    img = deskew(img)
    img = contrast_adjustment(img)
    img = remove_stamp(img)

    # Load sign detection models
    detection = torch.hub.load(
        'yolov5',
        'custom',
        source='local',
        path = 'model/best.pt',
        force_reload = True)
    
    # Load sign cleaner model
    cleaner_model_path = "signver/models/cleaner/small"
    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)
    img = signature_remove(img, detection, cleaner)

    # Load text detection model
    net = CRAFT()
    craft_model = 'model/craft_mlt_25k.pth'
    net.load_state_dict(copyStateDict(torch.load(craft_model, map_location='cpu')))
    net.eval()
    img = remove_background(img, net)


    cv2.imwrite('result/preprocess/test{}-preprocess.jpg'.format(img_idx), img)
