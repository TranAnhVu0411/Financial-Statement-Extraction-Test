import cv2
import pytesseract
import numpy as np
from wand.image import Image

img_idx=1
grid=False
# skew image (wand library)
with Image(filename='table-extraction-old/image/table/test{}.jpg'.format(img_idx)) as img:
    img.deskew(0.4*img.quantum_range)
    rotated = np.array(img)

result = rotated.copy()

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

cv2.imwrite('table-extraction-old/image/preprocess/test{}.png'.format(img_idx), result)

pdf = pytesseract.image_to_pdf_or_hocr(result, lang='vie', extension='pdf', config='--psm 4')
with open('table-extraction-old/pdf/test{}.pdf'.format(img_idx), 'w+b') as f:
    f.write(pdf) # pdf type is bytes by default