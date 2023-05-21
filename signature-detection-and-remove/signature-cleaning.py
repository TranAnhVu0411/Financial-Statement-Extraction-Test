import sys
sys.path.insert(0, 'signver')

from signver.cleaner import Cleaner
from signver.utils.data_utils import resnet_preprocess

import cv2
import numpy as np
import os

# Load clean model
cleaner_model_path = "signver/models/cleaner/small"
cleaner = Cleaner()
cleaner.load(cleaner_model_path)

# Read signatures image
signatures_path = 'signature-crop'
signatures = []
for i in os.listdir(signatures_path):
    signatures.append(cv2.imread(os.path.join(signatures_path, i)))

# Feature extraction with resnet model
sigs= [ resnet_preprocess( x, resnet=False, invert_input=False ) for x in signatures ]

# Normalization and clean
norm_sigs = [ x * (1./255) for x in sigs]
cleaned_sigs = cleaner.clean(np.array(norm_sigs))

# Reverse normalization
rev_norm_sigs = [ x / (1./255) for x in cleaned_sigs]

# Resize to the original size and save
signatures_clean_path = 'signature-clean'
for i in range(len(rev_norm_sigs)):
    cv2.imwrite(
        os.path.join(signatures_clean_path, 'sign_{}.jpg'.format(i)),
        cv2.resize(
            rev_norm_sigs[i],
            (signatures[i].shape[1], signatures[i].shape[0]),
            interpolation = cv2.INTER_CUBIC
        )
    )