import cv2
import matplotlib.pyplot as plt

def make_mask(image, low_threshold=(0, 0, 250), high_threshold=(255, 255, 255)):
    frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, low_threshold, high_threshold
    )
    return frame_threshold


def show_image(img):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()