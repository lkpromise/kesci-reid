## 对再辨识结果进行可视化的一个脚本--根据生成的json文件
## author：liu
## date:2019-10-28 20：30
import json
import os
import cv2
import numpy as np
query_a = "/home/liuk/data/kesci/test_data/query_a"
gallery_a = "/home/liuk/data/kesci/test_data/gallery_a"

with open("/home/liuk/code/kesci/kesci-master/result_1029.json",'r') as f:
    file = json.load(f)
    img_result = np.zeros((2560,1280,3),np.uint8)
    i = 0
    for key in file:
        if(i<10):
            query_path = os.path.join(query_a,key)
            img_result[i*256:(i+1)*256,0:128] = cv2.imread(query_path)
            for j,gallery in enumerate(file[key][:9]):
                gallery_path = os.path.join(gallery_a,gallery)
                img_result[i*256:(i+1)*256,(j+1)*128:(j+2)*128] = cv2.imread(gallery_path)
            i+=1
        else:
            break

    cv2.imwrite("result_1029.png",img_result)
