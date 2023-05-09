import cv2
import numpy as np
from wand.image import Image as wand_image
from PIL import Image, ImageEnhance
# import sys
# sys.path.insert(0, 'sign_detection')
from sign_detection.loader import make_mask, show_image
from sign_detection.extract import extract
from sign_detection.crop import run
from sign_detection.judge import judge

# Use cv2 image format

img_idx = 6
print('DESKEW')
# Deskew
with wand_image(filename='preprocess-document/image/test{}.jpg'.format(img_idx)) as img:
    img.deskew(0.4*img.quantum_range)
    deskew = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imwrite('preprocess-document/result/test{}deskew.jpg'.format(img_idx), deskew)

print('INCREASE CONTRAST')
# Increase contrast
im = Image.fromarray(deskew)
enhancer = ImageEnhance.Contrast(im)
factor = 1.5 #increase contrast
contrast = np.array(enhancer.enhance(factor))
cv2.imwrite('preprocess-document/result/test{}contrast.jpg'.format(img_idx), contrast)

print('REMOVE STAMP')
# Remove stamp
remove_stamp = contrast.copy()
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)
# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(contrast, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([155,25,0])
upper_red = np.array([179,255,255])
mask1 = cv2.inRange(contrast, lower_red, upper_red)

# join masks
mask = mask0+mask1

# lower = np.array([155,25,0])
# upper = np.array([179,255,255])
# mask = cv2.inRange(contrast, lower, upper)

remove_stamp = (remove_stamp*(np.expand_dims(cv2.bitwise_not(mask)/255, axis=2))+np.expand_dims(mask, axis=2)).astype('uint8')
cv2.imwrite('preprocess-document/result/test{}removestamp.jpg'.format(img_idx), remove_stamp)

print('REMOVE SIGNATURE')
# Remove signature
remove_signature = remove_stamp.copy()
mask = make_mask(remove_stamp)
labeled_mask = extract(mask, amplfier=15)
results = run(labeled_mask)
signatures = []
for key, value in results.items():
  result = judge(value['cropped_mask'])
  if result:
    print(key)
    signatures.append(value)

for idx, i in enumerate(signatures):
    region = i['cropped_region']
    remove_signature = cv2.rectangle(remove_signature, (region[0], region[1]), (region[0]+region[2], region[1]+region[3]), (255, 255, 255), -1)
    cv2.imwrite('preprocess-document/signature/test{}-{}.jpg'.format(img_idx, idx), remove_stamp[region[1]:region[1]+region[3], region[0]:region[0]+region[2]])
    cv2.imwrite('preprocess-document/signature/test{}-{}-mask.jpg'.format(img_idx, idx), i['cropped_mask'])

cv2.imwrite('preprocess-document/result/test{}removesignature.jpg'.format(img_idx), remove_signature)