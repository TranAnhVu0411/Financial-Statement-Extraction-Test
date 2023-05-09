# import module
from pdf2image import convert_from_path
import os
from tqdm import tqdm

# Store Pdf with convert_from_path function
for i in tqdm(os.listdir('pdf')):
    images = convert_from_path(os.path.join('pdf', i), poppler_path = r"poppler-23.01.0\Library\bin")
    image_path = os.path.join('image', os.path.splitext(i)[0])
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for j in range(len(images)):
        # Save pages as images in the pdf
        images[j].save(os.path.join(image_path, '{}.jpg'.format(j)), 'JPEG')