import sys
sys.path.insert(0, 'CRAFT_pytorch')

from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg
from CRAFT_pytorch.craft_utils import getDetBoxes, adjustResultCoordinates
from CRAFT_pytorch.test import copyStateDict
import torch
from torch.autograd import Variable
import numpy as np
import cv2

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

net = CRAFT()
net.load_state_dict(copyStateDict(torch.load('model/craft_mlt_25k.pth', map_location='cpu')))
net.eval()

refine_net = None

img_idx = 5
img_path = '../signature-detection-and-remove/result/preprocess/test{}removestamp-result.png'.format(img_idx)

img = cv2.imread(img_path)
if img.shape[0] == 2: img = img[0]
if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
if img.shape[2] == 4:   img = img[:,:,:3]
img = np.array(img)

text_threshold=0.7
link_threshold=0.4
low_text = 0.4
poly = False

bboxes, polys, score_text = test_net(net, img, text_threshold, link_threshold, low_text, poly, refine_net)
with open('metadata/word_meta{}.npy'.format(img_idx), 'wb') as f:
    np.save(f, bboxes)