from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from sklearn.model_selection import train_test_split


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

    @staticmethod
    def get_data_loader(data_x, data_y, is_train):
        dataset = SpeechDataset(data_x, data_y)
        if is_train:
            batch_size = 64  # TODO: Put this in config
            num_sample_per_epoch = 32
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
        # TODO: Add to config
        test_size = 0.2
        random_state = 123

        # Data Handling
        train_y = train_y - 1  # TODO: Handle this train_y -1 more beautifully
        train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y,
                                                                    test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=train_y)

        return train_x, validate_x, train_y, validate_y
