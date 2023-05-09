from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2

config = Cfg.load_config_from_name('vgg_transformer')

config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)
path = '/Users/trananhvu/Downloads/word (8).png'
img = cv2.imread(path)
img_pil = Image.fromarray(img)
s = detector.predict(img_pil)
print('text: ',s)