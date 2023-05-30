import cv2
from loader import make_mask, show_image
from extract import extract
from crop import run
from judge import judge

image = cv2.imread('/Users/trananhvu/Desktop/test/test6removestamp.jpg')
mask = make_mask(image)
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
    cv2.imwrite('image/sign/{}.jpg'.format(idx), image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]])