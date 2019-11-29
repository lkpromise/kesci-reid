## 一份可以参考的进行模型效果评测的代码
from dataset.data import read_image  # 图片读取方法，可以自己写，我是用的baseline里自带的
import os
import torch
import numpy as np
import json
from evaluate import eval_func, euclidean_dist, re_rank #  计算距离以及rerank，均是来自baseline

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1' # 指定gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference_samples(model, transform, batch_size): # 传入模型，数据预处理方法，batch_size
    query_list = list()
    with open(r'初赛A榜测试集/query_a_list.txt', 'r') as f:
                # 测试集中txt文件
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            img_file = os.path.join(r'初赛A榜测试集\query_a', image_name)  # 测试集query文件夹
            query_list.append(img_file)

    gallery_list = [os.path.join(r'初赛A榜测试集\gallery_a', x) for x in # 测试集gallery文件夹
                    os.listdir(r'初赛A榜测试集\gallery_a')]
    query_num = len(query_list)
    img_list = list()
    for q_img in query_list:
        q_img = read_image(q_img)
        q_img = transform(q_img)
        img_list.append(q_img)
    for g_img in gallery_list:
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)
    img_data = torch.Tensor([t.numpy() for t in img_list])
    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    for i in range(iter_n):
        print("batch ----%d----" % (i))
        batch_data = img_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            batch_feature = model(batch_data).detach().cpu()
            all_feature.append(batch_feature)
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]

    distmat = re_rank(query_feat, gallery_feat) # rerank方法
    distmat = distmat # 如果使用 euclidean_dist，不使用rerank改为：distamt = distamt.numpy()
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_200_indices = indices[:, :200]

    res_dict = dict()
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("\\")+1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("\\")+1:] for i in max_200_indices[q_idx]]
        res_dict[filename] = max_200_files

    with open(r'submission_A.json', 'w' ,encoding='utf-8') as f: # 提交文件
        json.dump(res_dict, f)
