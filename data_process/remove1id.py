'''
针对只有一个样本的训练文件进行移除

'''

import os
import shutil
from tqdm import tqdm
data_root = "/home/liuk/data/kesci/train_data/train_set"
## 建立数据的字典
id_dict = {}
print("开始建立字典...")
with open("/home/liuk/data/kesci/train_data/train_set/list.txt",'r') as f:
    files = f.readlines()
    for line in tqdm(files):
        id = line.split()[-1].strip()
        if id not in id_dict:
            id_dict[id] = []
        name = line.split()[0].split('/')[-1]
        id_dict[id].append(name)


## 统计每个id的数目，如果是一个则不进行操作，对于多于一个的再移动到另一个文件夹中作为新的训练数据

print("开始进行文件的移动操作...")
list_txt = open("/home/liuk/data/kesci/train_data/train_all_more1/list.txt",'a')
for id in tqdm(id_dict):
    pic = id_dict[id]
    if len(pic)==1:
        pass
    else:
        for i in pic:
            shutil.copyfile(os.path.join(data_root,i),os.path.join("/home/liuk/data/kesci/train_data/train_all_more1",i))
            list_txt.write("train/%s"%i+" "+id+"\n")


