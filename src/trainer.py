import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from torchaudio import transforms

from data_handler import DataHandler
from models import LSTM, CnnLSTM, CnnLSTMV2, CnnMaxPool
from sti_config import config


class Trainer:
    def __init__(self):
        # Set Seed
        seed = config.getint('TORCH', 'manual_seed')
        torch.manual_seed(seed)

        # Set default attributes
        self.dataloader = DataHandler()
        self.model = self.get_model()
        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.softmax = nn.Softmax(dim=1)
        self.epoch = config.getint('TRAINER', 'epoch')
        self.early_stop_freq = config.getint('TRAINER', 'early_stop_freq')

        # Set attributes that will update during training
        self.trained_time = None
        self.trained_epoch = None

    def train(self, train_x, train_y, validate_x, validate_y):
        logger.info('Training model')

        # Obtain train_set_dataloader and validation set data (Type: torch)
        train_set_dataloader, validate_x, validate_y = self.dataloader.prepare_training_data(train_x, train_y,
                                                                                             validate_x, validate_y)

        # Variables for early stop mechanism
        lowest_val_loss, lowest_val_loss_epoch, lowest_val_loss_time, early_stop_counter = 9999, 0, 0, 0
        lowest_val_loss_model = pickle.loads(pickle.dumps(self.model.state_dict()))

        train_start_time = time.time()
        for e in range(self.epoch):
            list_of_train_loss = list()  # Logging train loss
            for idx, (sampled_x, sampled_y) in enumerate(train_set_dataloader):
                # Data augmentation
                enable_augmentation = config.get('TRAINER', 'enable_augmentation')
                sampled_x = self.data_augmentation(sampled_x) if enable_augmentation else sampled_x

                # Reset gradient
                self.optimizer.zero_grad()

                # Get prediction
                y_pred = self.model(sampled_x)

                # Calculate loss
                train_loss = self.loss_function(y_pred, sampled_y)

                # Backpropagation
                train_loss.backward()
                self.optimizer.step()

                # Update epoch-wise train loss
                list_of_train_loss.append(train_loss.detach().numpy().mean())

            # Summarize the Train & Validation Loss
            with torch.no_grad():
                y_pred = self.model(validate_x)
                validate_loss = self.loss_function(y_pred, validate_y)
                avg_val_loss = validate_loss.mean()
                avg_train_loss = np.mean(list_of_train_loss)
                logger.info(f'epoch: {e}, train loss: {avg_train_loss}, validate loss: {avg_val_loss}')

            # Early stop mechanism
            early_stop_counter += 1
            if avg_val_loss < lowest_val_loss:
                lowest_val_loss = avg_val_loss
                lowest_val_loss_epoch = e
                lowest_val_loss_time = time.time()
                lowest_val_loss_model = pickle.loads(pickle.dumps(self.model.state_dict()))
                early_stop_counter = 0

            if early_stop_counter == self.early_stop_freq:
                logger.info(f'Early stop since validation loss had not drop for {self.early_stop_freq} epochs')
                break

        # Use the model with lowest validation loss for inference
        self.trained_time = round((lowest_val_loss_time - train_start_time) / 60)
        self.trained_epoch = lowest_val_loss_epoch
        logger.info(f'Lowest validation loss {lowest_val_loss} '
                    f'at Epoch: {lowest_val_loss_epoch} '
                    f'after {self.trained_time} minutes')
        logger.info(f'Replacing the trained model with the lowest validation loss model during training')
        self.model.load_state_dict(lowest_val_loss_model)

    def inference(self, train_x):
        train_x = torch.FloatTensor(train_x)
        y_pred = self.model(train_x)
        predicted_class = np.argmax(y_pred.detach().numpy(), axis=1)
        probability_per_class = self.softmax(y_pred).detach().numpy()
        return predicted_class, probability_per_class

    @staticmethod
    def get_model():
        method = config.get('TRAINER', 'model')
        logger.info(f"Model Used: {method}")

        if method == 'LSTM':
            mod = LSTM()
        elif method == 'CnnLSTM':
            mod = CnnLSTM()
        elif method == 'CnnLSTMV2':
            mod = CnnLSTMV2()
        elif method == 'CnnMaxPool':
            mod = CnnMaxPool()
        else:
            raise Exception
        return mod

    @staticmethod
    def get_loss_function():
        method = config.get('TRAINER', 'loss_function')
        logger.info(f"Loss function used: {method}")

        if method == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss()
        else:
            raise Exception
        return loss_function

    def get_optimizer(self):
        method = config.get('TRAINER', 'optimizer')
        learning_rate = config.getfloat('TRAINER', 'learning_rate')
        logger.info(f"Optimizer Used: {method} with learning rate: {learning_rate}")

        model_params = self.model.parameters()
        if method == 'adam':
            optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        else:
            raise Exception
        return optimizer

    @staticmethod
    def data_augmentation(data_x):
        # Augmentation via torchaudio
        mean = data_x.mean()
        data_x = data_x.swapaxes(1, 2)  # Reshape data into (N, F, T)
        # data_x = transforms.TimeStretch(freq_mask_param=6)(data_x, mask_value=mean)
        for _ in range(5):
            data_x = transforms.TimeMasking(time_mask_param=6)(data_x, mask_value=mean)
        for _ in range(2):
            data_x = transforms.FrequencyMasking(freq_mask_param=2)(data_x, mask_value=mean)
        data_x = data_x.swapaxes(1, 2)  # Reshape data back to (N, T, F)

        return data_x
