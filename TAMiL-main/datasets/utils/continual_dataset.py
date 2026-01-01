
from abc import abstractmethod
from argparse import Namespace
import torch
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np

class MalwareDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=None, label_path=None, indices=None, transform=None):
        # 기존:
        # self.data = np.load(data_path)
        # self.labels = np.load(label_path)
        # 수정 버전
        if data_path is not None and label_path is not None:
            self.data = np.load(data_path)
            self.labels = np.load(label_path)
        else:
            # 아래 코드에서 data/labels를 나중에 할당 가능하도록 일단 None
            self.data = None
            self.labels = None
        if indices is not None and self.data is not None:
            self.data = self.data[indices]
            self.labels = self.labels[indices]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x).float(), int(y)

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

def store_masked_loaders(train_full_dataset, test_full_dataset, setting):
    import numpy as np
    from torch.utils.data import DataLoader

    task_id = getattr(setting, 'current_task', None)
    n_tasks = setting.N_TASKS
    n_cls_per_task = setting.N_CLASSES_PER_TASK
    class_order = setting.class_order  # np.array or list of length 100

    # task별 start/end 인덱스 계산
    cls_start = 0 if task_id == 0 else 50 + (task_id - 1) * 5
    cls_end = 49 if task_id == 0 else cls_start + 4  # 각 task에 실제 class가 몇 개인지에 맞게

    # 이번 task에서 새롭게 배정된 클래스 인덱스
    if task_id == 0:
        train_classes = class_order[0:50]         # [0~49] (50개)
        test_classes = class_order[0:50]
    else:
        train_classes = class_order[cls_start:cls_end+1]   # 5개
        test_classes = class_order[0:cls_end+1]            # 누적 테스트

    # ===================== #
    # 마스킹 생성 (train_mask는 "이번 task 클래스만", test_mask는 "0~현재까지 누적 클래스")
    y_train = train_full_dataset.labels
    y_test = test_full_dataset.labels

    # 학습용: 현재 task의 새 클래스에 속한 샘플만 True
    train_mask = np.isin(y_train, train_classes)
    # 테스트용: 지금까지 등장한 전체 클래스에 속한 샘플만 True
    test_mask = np.isin(y_test, test_classes)

    # 마스킹해서 dataset 만듦
    from copy import deepcopy
    from .continual_dataset import MalwareDataset  # 혹은 해당 데이터셋명

    train_dataset_task = deepcopy(train_full_dataset)
    train_dataset_task.data = train_full_dataset.data[train_mask]
    train_dataset_task.labels = train_full_dataset.labels[train_mask]

    test_dataset_task = deepcopy(test_full_dataset)
    test_dataset_task.data = test_full_dataset.data[test_mask]
    test_dataset_task.labels = test_full_dataset.labels[test_mask]

    # DataLoader 생성
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset_task, batch_size=setting.args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset_task, batch_size=setting.args.batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader

# def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
#                     setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
#     """
#     Divides the dataset into tasks.
#     :param train_dataset: train dataset
#     :param test_dataset: test dataset
#     :param setting: continual learning setting
#     :return: train and test loaders
#     """
#     # core50 (i.e. uneven # of classes)
#     if type(setting.N_CLASSES_PER_TASK) == list:
#         FROM_CLASS = np.sum(setting.N_CLASSES_PER_TASK[:setting.i])
#         TO_CLASS = np.sum(setting.N_CLASSES_PER_TASK[:setting.i+1])
#         train_mask = np.logical_and(np.array(train_dataset.targets % 1000) >= FROM_CLASS,
#                                     np.array(train_dataset.targets % 1000) < TO_CLASS)
#         test_mask = np.logical_and(np.array(test_dataset.targets % 1000) >= FROM_CLASS,
#                                    np.array(test_dataset.targets % 1000) < TO_CLASS)
#         setting.i += 1
#     # any other dataset
#     else:
#         train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
#                                     np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
#         test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
#                                    np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

#         setting.i += setting.N_CLASSES_PER_TASK

#     train_dataset.data = train_dataset.data[train_mask]
#     test_dataset.data = test_dataset.data[test_mask]

#     train_dataset.targets = np.array(train_dataset.targets)[train_mask]
#     test_dataset.targets = np.array(test_dataset.targets)[test_mask]

#     train_loader = DataLoader(train_dataset,
#                               batch_size=setting.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset,
#                              batch_size=setting.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     setting.test_loaders.append(test_loader)
#     setting.train_loader = train_loader

#     return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    # core50 (i.e. uneven # of classes)
    if type(setting.N_CLASSES_PER_TASK) == list:
        FROM_CLASS = np.sum(setting.N_CLASSES_PER_TASK[:setting.i - 1])
        TO_CLASS = np.sum(setting.N_CLASSES_PER_TASK[:setting.i])
    # any other dataset
    else:
        FROM_CLASS = (setting.i - 1) * setting.N_CLASSES_PER_TASK
        TO_CLASS = (setting.i) * setting.N_CLASSES_PER_TASK

    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                FROM_CLASS, np.array(train_dataset.targets)
                                < TO_CLASS)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)