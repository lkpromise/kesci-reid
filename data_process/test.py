
import cv2
import numpy as np
data_pth_1 = "/home/liuk/data/kesci/train_data/train_expand/00073233b.png"
data_pth_2 = "/home/liuk/data/kesci/train_data/train_expand/00073233.png"

img1 = cv2.imread(data_pth_1)
img2 = cv2.imread(data_pth_2)

print(np.sum(img1-img2))
