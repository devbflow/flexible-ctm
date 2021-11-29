import os
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd


class MiniImagenetDataset(Dataset):
    """Dataset subclass for miniImagenet."""
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.labels = pd.read_csv(annotations_file)
        self.encoding = self._encode_labels()
        self.img_dir = img_dir
        if transform:
            self.transform = transform
        else:
            size = 224
            # transform.ToTensor not necessary
            self.transform = transforms.Compose([transforms.Resize((size,size)),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = self._get_encoding

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # if we get a list of indices, get several samples
        if type(idx) == list:
            imgs = []
            labels = []
            for i in idx:
                img_path = os.path.join(self.img_dir, self.labels.iat[i, 0])
                img = read_image(img_path).float()
                label = self.labels.iat[i, 1]
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    label = self.target_transform(label)
                imgs.append(img.unsqueeze(dim=0))
                labels.append(label)
            imgs_tensor = torch.cat(imgs, dim=0)
            return imgs_tensor, labels
        else:
            # we only have one sample to load
            img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
            img = read_image(img_path).float()
            label = self.labels.iloc[idx,1]
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                label = self.target_transform(label)
            return img, label

    def _encode_labels(self):
        """Returns dictionary with original label mapped to integers."""
        labels = pd.unique(self.labels['label'])
        encoding = {}
        for i in range(labels.shape[0]):
            encoding[labels[i]] = i
        #print("Encoding: ", encoding)
        return encoding

    def _get_encoding(self, label):
        """Returns encoded version of label."""
        return self.encoding[label]


class FewShotBatchSampler(Sampler):

    def __init__(self, targets, n_way, k_shot, include_query=True, shuffle=True):
        #self.dataset = data_source
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
        self.cls_batchnum = {}
        # get indices for each class label and calculate resp. batch number in case of inequalities
        for c in self.classes:
            self.cls_indices[c] = self.labels.index[self.labels['label'] == c].tolist()
            self.cls_batchnum[c] = len(self.cls_indices[c]) // self.k
        self.iterations = sum(self.cls_batchnum.values()) // self.n
        self.class_list = [c for c in self.classes for _ in range(self.cls_batchnum[c])]

        if self.shuffle:
            self._shuffle_data()

    def __iter__(self):
        if self.shuffle:
            self._shuffle_data()

        start_idx = defaultdict(int)
        for i in range(self.iterations):
            cls_batch = self.class_list[i*self.n : (i+1)*self.n]
            idx_batch = []
            for c in cls_batch:
                idx_batch.extend(self.cls_indices[c][start_idx[c]:start_idx[c] + self.k])
                start_idx[c] += self.k

            #print("start_idx: ", start_idx)
            # yield support and query or only one set if include_query is set accordingly
            if self.include_query:
                yield idx_batch[::2] + idx_batch[1::2]
            else:
                yield idx_batch

    def __len__(self):
        return self.iterations

    def _shuffle_data(self):
        '''Shuffle data properly to not have repeating classes per N-way K-shot batch.'''
        self.class_list = [c for _ in range(self.iterations) for c in random.sample(self.class_list, self.n) ]
        for c in self.classes:
            random.shuffle(self.cls_indices[c])


def encode_all_labels(dataset_path, splits=['train', 'test', 'val']):
    """Function to encode all labels across all splits."""
    labels = []
    for s in splits:
        split_file = pd.read_csv(os.path.join(dataset_path, s+'.csv'))
        l = pd.unique(split_file['label'])
        for i in range(l.shape[0]):
            labels.append(l[i])
    encoded = {}
    for i, label in enumerate(labels):
        encoded[label] = i
    return encoded

def make_crossentropy_targets(supp_labels, query_labels, k_shot):
    """
    Transform query labels into valid cross-entropy targets.
    Changes labels from
    ..math::
        l_i \in \{0, \ldots, numDatasetClasses\}
        to
        l_i \in [0, C-1]
    where :math: C is the number of classes in the batch (i.e. n_way).
    """
    targets = torch.stack(
        [torch.nonzero(
            torch.eq(supp_labels[::k_shot], q)
        ) for q in query_labels]
    )
    return targets.squeeze()

DATASETS = {'miniImagenet': MiniImagenetDataset}

def split_support_query(batch, labels, device='none'):
    """Split up the combined batch of support and query set and their labels."""

    support_set, query_set = batch.chunk(2, dim=0)
    support_labels = labels[:len(labels)//2]
    query_labels = labels[len(labels)//2:]
    if device != 'none':
        support_set = support_set.to(device)
        query_set = query_set.to(device)
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)
    return support_set, query_set, support_labels, query_labels

def load_dataset(dataset_path, split='train'):
    """Returns a Dataset object for the specified dataset"""
    dataset_name = os.path.basename(dataset_path)
    annotations = os.path.join(dataset_path, split+'.csv')
    img_dir = os.path.join(dataset_path, 'images')
    return DATASETS[dataset_name](annotations, img_dir)

def get_dataloader(dataset_path, n_way, k_shot, include_query=True, shuffle=True, split='train', **kwargs):
    """Returs a DataLoader for the specified dataset according to passed args."""
    dataset = load_dataset(dataset_path, split)
    # one batch returned by sampler consists of N*K samples for support and/or query set
    batch_sampler = FewShotBatchSampler(targets=dataset.labels,
                                        n_way=n_way,
                                        k_shot=k_shot,
                                        shuffle=shuffle,
                                        include_query=include_query)

    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            batch_size=1,
                            **kwargs)
    return dataloader

if __name__ == "__main__":
    loader = get_dataloader(dataset_path='miniImagenet',
                            n_way=2,
                            k_shot=5,
                            include_query=True,
                            split='train',
                            shuffle=True)
    # loader check
    print("get_loader test...")
    c = 0
    for (batch, labels) in loader:
        if c == 1:
            break
        print("Batch:", batch.shape)
        print("Label:", labels)
        if loader.batch_sampler.include_query:
            support_set, query_set, support_labels, query_labels = split_support_query(batch, labels)
            print(support_set.shape)
            print(support_labels)
            print(query_set.shape)
            print(query_labels)
        c += 1
    print("...exit get_loader test.")

    #print("Enter encode_all_labels test...")
    #print(encode_all_labels(dataset_path='miniImagenet'))
    #print("...exit test")
