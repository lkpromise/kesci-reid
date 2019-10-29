## 查看数据分布的一个脚本
## author：liu
## date:2019-10-28
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

data_pth = "/home/liuk/data/kesci/train_data/train_set/train_list.txt"

## 返回id，id字典，number
def construct(data_pth):
    X_data = []
    X_data_sum = {}
    Y_data = []
    with open(data_pth,'r') as f:
        file = f.readlines()
        for line in file:
            id = int(line.split()[-1])
            if id not in X_data:
                X_data.append(int(id))
                X_data_sum[id] = 1
            else:
                X_data_sum[id] +=1
    for i in range(len(X_data)):
        Y_data.append(X_data_sum[i])
    return X_data,X_data_sum,Y_data

if __name__=="__main__":
    count = 0
    total_number = 0
    X_data,X_data_sum,Y_data = construct(data_pth)
    valid_id = []
    query_txt = open("/home/liuk/data/kesci/valid_data/query/query_list.txt",'a')
    gallery_txt = open("/home/liuk/data/kesci/valid_data/gallery/gallery_list.txt",'a')
    for key in X_data_sum:
        number = X_data_sum[key]
        if number>3 and number<=6:
            valid_id.append(key)
            # count +=1
            # total_number +=number
            #print(key,number)
    query_id = []
    train_cp = "/home/liuk/data/kesci/train_data/train_part"
    target_query = "/home/liuk/data/kesci/valid_data/query"
    target_gallery = "/home/liuk/data/kesci/valid_data/gallery"
    source_pth = "/home/liuk/data/kesci/train_data/train_set"
    with open(data_pth,'r') as f:
        file = f.readlines()
        for line in tqdm(file):
            id = int(line.split()[-1])
            img_name = line.split()[0].split('/')[-1]
            if (id in valid_id) and (id not in query_id):
                #shutil.move(os.path.join(train_cp,img_name),os.path.join(target_query,img_name))
                query_txt.write(line)
                query_id.append(id)
            elif (id in valid_id) and (id in query_id):
                #shutil.move(os.path.join(train_cp,img_name),os.path.join(target_gallery,img_name))
                gallery_txt.write(line)
            else:
                pass


    ## 处理部分训练集的文本部分

    file_name = os.listdir(train_cp)
    ## 构建存在图片名列表
    valid_list = []
    for i in file_name:
        if i.split('.')[-1]=="png":
            valid_list.append(i)
        else:
            pass
    train_pt = open("/home/liuk/data/kesci/train_data/train_part/train_valid.txt",'a')
    with open(data_pth,'r') as f:
        file = f.readlines()
        for line in file:
            img_name = line.split()[0].split('/')[-1]
            if img_name in valid_list:
                train_pt.write(line)


    #print("count={},total_number = {}".format(count,total_number))
    ##  count=1570,total_number = 8719
