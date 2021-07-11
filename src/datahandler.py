import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler

from sti_config import config


class SpeechDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        data_x = self.data_x[idx]
        data_y = self.data_y[idx]
        return data_x, data_y


class DataHandler:
    def __init__(self):
        pass

    def preprocess_data(self, train_x, train_y):
        train_x, validate_x, train_y, validate_y = self.train_test_split(train_x, train_y)
        train_x, train_y = self.data_augmentation(train_x, train_y)
        train_set_dataloader = self.get_data_loader(train_x, train_y, is_train=True)
        validate_x = torch.FloatTensor(validate_x)
        validate_y = torch.LongTensor(validate_y)
        return train_set_dataloader, validate_x, validate_y

    @staticmethod
    def get_data_loader(data_x, data_y, is_train):
        dataset = SpeechDataset(data_x, data_y)
        if is_train:
            batch_size = config.getint('TRAINING', 'batch_size')
            num_sample_per_epoch = config.getint('TRAINING', 'num_sample_per_epoch')
            random_sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size*num_sample_per_epoch)
            batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)
            dataloader = DataLoader(dataset=dataset, batch_size=None, sampler=batch_sampler)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=dataset.__len__())
        return dataloader

    @staticmethod
    def data_augmentation(data_x, data_y):
        # TODO: add code to support data augmentation
        return data_x, data_y

    @staticmethod
    def train_test_split(train_x, train_y):
        test_size = config.getfloat('TRAIN_VALIDATION_SPLIT', 'validate_size')
        random_state = config.getint('TRAIN_VALIDATION_SPLIT', 'random_state')

        # Data Handling
        train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y,
                                                                    test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=train_y)

        return train_x, validate_x, train_y, validate_y
