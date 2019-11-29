## 用来对数据进行增强，扩展规则如下：
## 1、对于少于六张的图片，运用cutout的形式将其扩充为6张
## 2、对于多于六张的图片，随机筛选出六张

import os
import math
import random
import torch
import numpy as np
from tqdm import tqdm
import shutil
# from PIL import Image
import cv2

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        #print(type(mask))
        img = img * mask

        return img
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def int_random(a,n):
    ## 一个数组存储生成的随机数
    a_list = []
    while len(a_list)<n:
        d_int = random.randint(0,a-1)
        if d_int not in a_list:
            a_list.append(d_int)
        else:
            pass
    return a_list

def expand_data(raw_folder,tar_data):
    raw_data = os.path.join(raw_folder,"list.txt")
    id_dict = {}
    add_name = ['a','b','c','d','e','f']
    with open(raw_data,'r') as f:
        file = f.readlines()
        for line in file:
            id = line.split()[-1]
            name = line.split()[0]
            if id not in id_dict:
                id_dict[id] = [name,]
            else:
                id_dict[id].append(name)
    new_file_txt = open(os.path.join(tar_data,"list.txt"),'a')
    print("正在生成图片...")
    for i in tqdm(id_dict):
        file = id_dict[i]
        number = len(file)
        count = number
        if count<=6:
            ## 把原始的图片移入目标文件夹
            for temp in file:
                pic_name = temp.split('/')[-1]
                shutil.copyfile(os.path.join(raw_folder,pic_name),os.path.join(tar_data,pic_name))
            while(count<=5):
                j = random.randint(0,number-1)
                #print(j)
                #print(file[j])
                name = file[j].split('/')[-1]
                img = cv2.imread(os.path.join(raw_folder,name))
                img = torch.from_numpy(img).float()
                # print(type(img))
                ## 随机消除
                #erasing=RandomErasing(1.0)
                ## cutout
                erasing = Cutout(1,32)
                img = erasing(img)
                img = img.numpy()
                # print(type(img))
                ## 生成新的名字
                # m = random.randint(0,5)
                new_name = name.split('.')[0]+add_name[count]+".png"
                ## 写入新的图片到目标地
                tar_path = os.path.join(tar_data,new_name)
                # img.save(tar_path)
                cv2.imwrite(tar_path,img)
                ## 把新的图片名字写入字典
                s_name = os.path.join("train",new_name)
                id_dict[i].append(s_name)
                count +=1
        else:
            # 这是多余6张的只采样六张，感觉没有充分利用数据
            # chose_id = int_random(count,6)
            # new_file = []
            # for id in chose_id:
            #     file_name = file[id]
            #     new_file.append(file_name)
            # id_dict[i] = new_file

            ## 将图片拷贝入目标文件夹
            for pic in id_dict[i]:
                pic_name = pic.split('/')[-1]
                shutil.copyfile(os.path.join(raw_folder,pic_name),os.path.join(tar_data,pic_name))
    ## 写新的list.txt文件
    print("正在写入新的txt文件...")

    for i in tqdm(id_dict):
        id = i
        for pic in id_dict[id]:
            new_file_txt.write(pic+" "+id+'\n')
        # print(id_dict[i])
if __name__ == '__main__':
    raw_data = "/home/liuk/data/kesci/train_data/train_part"
    tar_data = "/home/liuk/data/kesci/train_data/train_cut"
    expand_data(raw_data,tar_data)
