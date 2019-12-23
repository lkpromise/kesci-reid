# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import random

import os
import os.path as osp

from .bases import BaseImageDataset


class Kesci(BaseImageDataset):

    dataset_dir = 'kesci-fu/train_valid'

    def __init__(self, root='/home/liuk/data/', verbose=True, **kwargs):
        super(Kesci, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        ## 用全部数据训练，并生成测试结果
        # self.train_dir = osp.join(self.dataset_dir, 'train_data/train_set')
        # self.query_dir = osp.join(self.dataset_dir, 'test_data/query_a')
        # self.gallery_dir = osp.join(self.dataset_dir, 'test_data/gallery_a')

        self.train_dir = osp.join(self.dataset_dir, 'train_temp')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True,type=True)
        query = self._process_dir(self.query_dir, relabel=True)
        gallery = self._process_dir(self.gallery_dir, relabel=True)

        # train = self._process_dir(self.train_dir, relabel=True,type=True)
        # query = self._process_dir(self.query_dir)
        # gallery = self._process_dir(self.gallery_dir)

        if verbose:
            print("=> Kesci loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False,type=False,gan=False):
        dataset = []
        if relabel:
            file_path = osp.join(dir_path,"list.txt")
            pid_container = set()
            with open(file_path,'r') as f:
                file = f.readlines()
                for line in file:
                    id = int(line.split()[-1])
                    pid_container.add(id)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            with open(file_path,'r') as f:
                file = f.readlines()
                for line in file:
                    if gan:
                        img_name = line.split()[0]
                    else:
                        img_name = line.split()[0].split('/')[-1]
                    img_path = osp.join(dir_path,img_name)
                    id = int(line.split()[-1])
                    if type: id =pid2label[id]
                    camid = random.randint(0,6)
                    dataset.append((img_path,id,camid))
        else:
            img_name = os.listdir(dir_path)
            for i in img_name:
                img_path = osp.join(dir_path,i)
                id = i.split('.')[0]
                # id = random.randint(4768,8600)
                camid = random.randint(0,6)
                dataset.append((img_path,id,camid))
        return dataset




        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if relabel: pid = pid2label[pid]
        #     dataset.append((img_path, pid, camid))
