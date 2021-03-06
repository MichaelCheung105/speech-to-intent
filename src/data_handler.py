import numpy as np
import torch
from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler

from sti_config import HOME_PATH, config


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

    def prepare_training_data(self, train_x, train_y, validate_x, validate_y):
        train_set_dataloader = self.get_data_loader(train_x, train_y, is_train=True)
        validate_x = torch.FloatTensor(validate_x)
        validate_y = torch.LongTensor(validate_y)
        return train_set_dataloader, validate_x, validate_y

    @staticmethod
    def get_data_loader(data_x, data_y, is_train):
        dataset = SpeechDataset(data_x, data_y)
        if is_train:
            batch_size = config.getint('TRAINER', 'batch_size')
            num_batch_per_epoch = config.getint('TRAINER', 'num_batch_per_epoch')
            random_sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size*num_batch_per_epoch)
            batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)
            dataloader = DataLoader(dataset=dataset, batch_size=None, sampler=batch_sampler)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=dataset.__len__())
        return dataloader

    @staticmethod
    def train_test_split(train_x, train_y, train_i):
        logger.info('Applying TRAIN-VALIDATION split')

        # Config
        test_size = config.getfloat('TRAIN_VALIDATION_SPLIT', 'validate_size')
        random_state = config.getint('TRAIN_VALIDATION_SPLIT', 'random_state')

        # Data Handling
        train_x, validate_x, train_y, validate_y, train_i, validate_i = train_test_split(train_x, train_y, train_i,
                                                                                         test_size=test_size,
                                                                                         random_state=random_state,
                                                                                         stratify=train_y)

        return train_x, validate_x, train_y, validate_y, train_i, validate_i

    @staticmethod
    def read_data(data_source):
        logger.info(f'Loading {data_source} data')

        data_path = config.get('DEFAULT', 'data_dir')
        file_name = 'data_for_training' if data_source == 'train' else 'test_data'
        data = np.load(f"{HOME_PATH}/{data_path}/{file_name}.npz")
        samples = data['f0']
        labels = data['f1'] - 1  # -1 to make the classes starts with 0
        indices = np.array(range(samples.shape[0]))
        return samples, labels, indices
