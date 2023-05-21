import cv2
import torch
import os

# Load detection models
model = torch.hub.load(
    'yolov5',
    'custom',
    source='local',
    path = 'model/best.pt',
    force_reload = True)

img_path = '../preprocess-document/result-preprocess'
org_imgs = [] # Saving original images
imgs = [] # Saving YOLOv5 Result Images
for i in os.listdir(img_path):
    if 'removestamp' in i:
        img = cv2.imread(os.path.join(img_path, i))
        org_imgs.append(img.copy())
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV image (BGR to RGB), for saving YOLOv5 result

results = model(imgs, size=640)  # includes NMS
results.save()  # save as results1.jpg, results2.jpg... etc.


# Loop through image
signature_path = 'signature-crop'
for img_idx, img_result in enumerate(results.xyxy):
    # If there is no signature predict, continue
    image = org_imgs[img_idx]
    if len(img_result)==0:
        continue
    # Loop through signature
    for sign_idx, signature in enumerate(img_result):
        x0 = int(signature[0])
        y0 = int(signature[1])
        x1 = int(signature[2])
        y1 = int(signature[3])

        # Save crop signature
        cv2.imwrite(
            os.path.join(signature_path, 'test-{}-signature-{}.jpg'.format(img_idx, sign_idx)),
            cv2.cvtColor(
                image[y0:y1, x0:x1],
                cv2.COLOR_RGB2BGR)  # OpenCV image (BGR to RGB)
        )