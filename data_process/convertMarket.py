"""
转换当前数据集到market的格式，DG-Net的使用

"""

import os
import shutil
import random
from tqdm import tqdm
## query图片进行格式转换
## -------------------------------------------------
print("正在转换query图为market的格式...")
query_pth = "/home/liuk/data/kesci/valid_data/query"
count = 0
target_query = "/home/liuk/data/kesci/kesciMarket/query/"
with open("/home/liuk/data/kesci/valid_data/query/list.txt",'r') as f:
    file = f.readlines()
    for line in tqdm(file):
        id = int(line.split()[-1])
        name = line.split()[0].split('/')[-1]
        camer_id = random.randint(1,3)
        new_name = "%04d_c%ds%d_%06d_01.png"%(id,camer_id,camer_id,count)
        shutil.copyfile(os.path.join(query_pth,name),os.path.join(target_query,new_name))
        count+=1
print("目前的count值是：",count)
#----------------------------------------------------

## gallery图片进行格式转换
## -------------------------------------------------
print("正在转换gallery为market格式...")
gallery_pth = "/home/liuk/data/kesci/valid_data/gallery"
# count = 0
target_gallery = "/home/liuk/data/kesci/kesciMarket/bounding_box_test/"
with open("/home/liuk/data/kesci/valid_data/gallery/list.txt",'r') as f:
    file = f.readlines()
    for line in tqdm(file):
        id = int(line.split()[-1])
        name = line.split()[0].split('/')[-1]
        #print(name)
        camer_id = random.randint(1,3)
        new_name = "%04d_c%ds%d_%06d_01.png"%(id,camer_id,camer_id,count)
        shutil.copyfile(os.path.join(gallery_pth,name),os.path.join(target_gallery,new_name))
        count+=1
print("目前的count值是:",count)
#----------------------------------------------------
## train图片进行格式转换
## -------------------------------------------------
print("正在转换训练图片到market格式...")
train_pth = "/home/liuk/data/kesci/train_data/train_part"
target_train = "/home/liuk/data/kesci/kesciMarket/bounding_box_train/"
with open("/home/liuk/data/kesci/train_data/train_part/list.txt",'r') as f:
    file = f.readlines()
    for line in tqdm(file):
        id = int(line.split()[-1])
        name = line.split()[0].split('/')[-1]
        camer_id = random.randint(1,3)
        new_name = "%04d_c%ds%d_%06d_01.png"%(id,camer_id,camer_id,count)
        shutil.copyfile(os.path.join(train_pth,name),os.path.join(target_train,new_name))
        count+=1
print("已全部转换完成，目前的图片数量是:",count)
#----------------------------------------------------
