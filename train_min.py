import argparse
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
import clip

import train_new
from utils.data_processing import produce_labels
from utils.utils import int_item


DEFAULT_TRAIN_NUM = 100
DEFAULT_VAL_NUM = 20


class MinTextDataset(data.Dataset):
    def __init__(self, split="train", train_num=None, val_num=None):
        if train_num is None:
            train_num = DEFAULT_TRAIN_NUM
        if val_num is None:
            val_num = DEFAULT_VAL_NUM
        self.text_dir = "data/celeba-caption/"
        self.text_files = os.listdir(self.text_dir)
        self.text_files.sort(key=int_item)

        f = open("data/list_attr_celeba.txt")
        data_lines = f.readlines()
        attrs = data_lines[1].split(" ")
        attrs[-1] = attrs[-1][:-1]
        self.attrs = np.array([" ".join(a.split("_")).lower() for a in attrs], dtype=object)
        self.anno = data_lines[2:]

        if split == "train":
            self.text_files = self.text_files[:train_num]
            self.anno = self.anno[:train_num]
        else:
            self.test_latents = torch.load("data/test_latents_seed100.pt")
            val_num = min(val_num, self.test_latents.shape[0])
            self.text_files = self.text_files[train_num:train_num + val_num]
            self.anno = self.anno[train_num:train_num + val_num]

        self.split = split
        self.non_represents = ["no", "hair", "wearing", "eyebrows", "eyes", "big", "nose", "o"]
        self.gender_list = ["he", "she", "man", "woman"]

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, index):
        text_filename = self.text_files[index]
        text_path = os.path.join(self.text_dir, text_filename)
        text_set = open(text_path).readlines()

        sampled_text = text_set[0][:-1]
        anno = self.anno[index][:-1].split(" ")[1:]
        clip_text, labels, exist_mask = produce_labels(
            sampled_text,
            anno,
            self.attrs,
            self.gender_list,
            self.non_represents,
        )

        length = torch.where(clip_text == 0)[1][0].item()

        if self.split == "train":
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length
        return clip_text.squeeze(0), sampled_text, labels, exist_mask, length, self.test_latents[index]


class MinPartTextDataset(data.Dataset):
    def __init__(self, split="train", sample_num=3, train_num=None, val_num=None):
        if train_num is None:
            train_num = DEFAULT_TRAIN_NUM
        if val_num is None:
            val_num = DEFAULT_VAL_NUM
        self.test_latents = torch.load("data/test_latents_seed100.pt")
        val_num = min(val_num, self.test_latents.shape[0])
        self.split = split
        self.sample_num = sample_num

        f = open("data/list_attr_celeba.txt")
        self.data = f.readlines()
        attrs = self.data[1].split(" ")
        attrs[-1] = attrs[-1][:-1]
        self.attrs = np.array([" ".join(a.split("_")).lower() for a in attrs], dtype=object)

        if split == "train":
            self.img_attr = self.data[2:2 + train_num]
        else:
            start = 2 + train_num
            self.img_attr = self.data[start:start + val_num]

        self.hair = [
            "bald",
            "bangs",
            "black hair",
            "blond hair",
            "brown hair",
            "gray hair",
            "receding hairline",
            "straight hair",
            "wavy hair",
        ]
        self.eye = ["arched eyebrows", "bags under eyes", "bushy eyebrows", "eyeglasses", "narrow eyes"]
        self.fashion = [
            "attractive",
            "heavy makeup",
            "high cheekbones",
            "rosy cheeks",
            "wearing earrings",
            "wearing hat",
            "wearing lipstick",
            "wearing necklace",
            "wearing necktie",
        ]
        self.others = [
            "5 o clock shadow",
            "big nose",
            "blurry",
            "chubby",
            "double chin",
            "no beard",
            "oval face",
            "pale skin",
            "pointy nose",
            "young",
        ]
        self.mouth = ["big lips", "mouth slightly open", "smiling", "goatee", "mustache", "sideburns"]

        self.groups = [self.hair, self.eye, self.fashion, self.others, self.mouth]

    def __len__(self):
        return len(self.img_attr)

    def __getitem__(self, index):
        sampled_class = torch.randint(0, 5, (1,)).item()
        instance_attr = self.groups[sampled_class]
        sampled_cate = torch.randperm(len(instance_attr))[: self.sample_num]
        attr = np.array(instance_attr)[sampled_cate]
        if self.sample_num == 1:
            attr = np.array([attr])

        selected_cate_40 = []
        for x in attr:
            selected_cate_40.append(int(np.where(self.attrs == x)[0][0]))

        gender = torch.randint(0, 3, (1,)).item()
        concat_text = ", ".join(attr)
        if gender == 0:
            sampled_text = "she has " + concat_text
        elif gender == 1:
            sampled_text = "he has " + concat_text
        else:
            sampled_text = "the person has " + concat_text

        clip_text = clip.tokenize(sampled_text)

        exist_mask = torch.zeros(40, dtype=torch.float32)
        idx = torch.tensor(selected_cate_40, dtype=torch.long)
        exist_mask[idx] = 1.0
        labels = exist_mask.clone()

        if gender != 2:
            exist_mask[20] = 1
            if gender == 1:
                labels[20] = 1
            else:
                labels[20] = 0

        length = torch.where(clip_text == 0)[1][0].item()

        if self.split == "train":
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length
        return clip_text.squeeze(0), sampled_text, labels, exist_mask, length, self.test_latents[index]


def _ensure_arg(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def _consume_min_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train_num", type=int, default=DEFAULT_TRAIN_NUM)
    parser.add_argument("--val_num", type=int, default=DEFAULT_VAL_NUM)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    global DEFAULT_TRAIN_NUM, DEFAULT_VAL_NUM
    min_args = _consume_min_args()
    DEFAULT_TRAIN_NUM = min_args.train_num
    DEFAULT_VAL_NUM = min_args.val_num

    _ensure_arg("--train_num", DEFAULT_TRAIN_NUM)
    _ensure_arg("--val_num", DEFAULT_VAL_NUM)
    _ensure_arg("--epochs", 1)
    _ensure_arg("--batch-size", 1)
    _ensure_arg("--test_batch", 1)
    _ensure_arg("--workers", 0)
    _ensure_arg("--print-freq", 1)
    _ensure_arg("--loss_w_norm_weight", 0.0001)
    _ensure_arg("--loss_clip_weight", 0)
    _ensure_arg("--loss_face_bg_weight", 0)
    _ensure_arg("--loss_face_norm_weight", 0)
    _ensure_arg("--loss_id_weight", 0)
    _ensure_arg("--loss_minmaxentropy_weight", 0)

    train_new.TextDataset = MinTextDataset
    train_new.PartTextDataset = MinPartTextDataset
    train_new.main()


if __name__ == "__main__":
    main()
