import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import read_image
import pandas as pd


class MiniImagenetDataset(Dataset):
    """Dataset subclass for miniImagenet."""
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        img = read_image(img_path)
        label = self.labels.iloc[idx,1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

class FewShotBatchSampler(Sampler):

    def __init__(self, data_source, targets, n_way, k_shot, include_query=True, shuffle=True):
        self.dataset = data_source
        self.labels = targets
        self.n = n_way
        self.k = k_shot
        self.include_query = include_query
        if self.include_query:
            self.k *= 2
        self.shuffle = shuffle
        self.batch_size = self.n * self.k

        self.classes = pd.unique(targets['label'])
        self.num_classes = len(self.classes)
        self.cls_indices = {}
        self.cls_batches = {}
        # get indices for each class label and claculate resp. batch number in case of inequalities
        for c in self.classes:
            self.cls_indices[c] = self.labels.index[self.labels['label'] == c].tolist()
            self.cls_batches[c] = self.cls_indices[c].shape[0] // self.k
        self.iterations = sum(self.cls_batches.values()) // self.n
        self.class_list = [c for c in self.classes for _ in range(self.cls_batches[c])]

        if self.shuffle:
            self.labels = self.labels.sample(frac=1)
            #TODO: shuffle
    def __iter__(self):
        if self.shuffle:
            #TODO:shuffle
            pass

        start_idx = defaultdict(int)
        for i in range(self.iterations):
            cls_batch = self.class_list[i*self.n : (i+1)*self.n]
            idx_batch = []
            for c in cls_batch:
                idx_batch.extend(self.cls_indices[c][start_idx[c]:start_idx[c] + self.k])
                start_idx += self.k

            # yield support and query or only one set if include_query
            if self.include_query:
                support_set = idx_batch[::2]
                query_set = idx_batch[1::2]
                yield support_set, query_set
            else:
                yield idx_batch

    def __len__(self):
        return self.iterations

DATASETS = {'miniImagenet': MiniImagenetDataset}

def load_dataset(dataset_name, split='train'):
    """Returns a DataLoader for the specified dataset"""
    dataset_path = os.path.abspath(dataset_name)
    annotations = os.path.join(dataset_path, split+'.csv')
    img_dir = os.path.join(dataset_path, 'images')
    return DATASETS[dataset_name](annotations, img_dir)

def get_dataloader(dataset_name, n_way, k_shot, shuffle=True, split='train'):
    dataset = load_dataset(dataset_name, split)
    sampler = FewShotBatchSampler(data_source=dataset,
                                  targets=dataset.labels,
                                  n_way=n_way,
                                  k_shot=k_shot,
                                  shuffle=shuffle)

    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler)
    return dataloader
if __name__ == "__main__":
    pass
