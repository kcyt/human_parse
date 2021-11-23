#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   dataset.py
@Time    :   8/30/19 9:12 PM
@Desc    :   Dataset Definition
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import cv2
import numpy as np

from torch.utils import data
from utils.transforms import get_affine_transform


class SimpleFolderDataset(data.Dataset):
    def __init__(self, input_size=[512, 512], transform=None):
        #self.root = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_fixed_full_mesh"
        self.root = "/content/drive/MyDrive/buffer_fixed_full_mesh" # for gdrive

        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        #self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/train_set_list.txt", dtype=str).tolist()
        #self.training_subject_list = np.loadtxt("/content/drive/MyDrive/train_set_list.txt", dtype=str).tolist()  # for gdrive


        #self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/fake_train_set_list.txt", dtype=str).tolist()
        self.training_subject_list = np.loadtxt("/content/drive/MyDrive/fake_train_set_list.txt", dtype=str).tolist()  # for gdrive
        print("using fake training subject list!")

        #self.test_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/test_set_list.txt", dtype=str).tolist()
        self.test_subject_list = np.loadtxt("/content/drive/MyDrive/test_set_list.txt", dtype=str).tolist()


        self.subjects = self.training_subject_list # change to self.test_subject_list to get test subjects


        self.file_list = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.root, training_subject)
            subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
            self.file_list = self.file_list + subject_render_paths_list
        self.file_list = sorted(self.file_list)


    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        #img_name = self.file_list[index]
        #img_path = os.path.join(self.root, img_name)

        img_path = self.file_list[index]
        subject = img_path.split('/')[-2]
        img_name = os.path.join(subject,  os.path.splitext(os.path.basename(img_path))[0] )
        img_name = img_name.replace("image","parse")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'name': img_name,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta
