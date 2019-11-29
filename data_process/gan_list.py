'''
根据生成的数据，产生一个txt文件用于训练，看一下会不会有效果的提升。

'''

import os
from tqdm import tqdm

root_pth = "/home/liuk/data/kesci/train_data/synchronized_img"

second_pth = os.listdir(root_pth)

list_txt = open("/home/liuk/data/kesci/train_data/synchronized_img/list.txt",'a')
for i in second_pth:
    print("我正在处理%s的文件夹"%i)
    third = os.path.join(root_pth,i)
    third_pth = os.listdir(third)
    for j in tqdm(third_pth):
        pic_name = os.listdir(os.path.join(third,j))
        for name in pic_name:
            id = name.split('_')[0]
            list_txt.write(os.path.join(i,j,name)+" "+id+"\n")

