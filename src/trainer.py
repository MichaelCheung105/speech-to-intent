import pickle

import numpy as np
import torch
import torch.nn as nn
from logzero import logger

from datahandler import DataHandler
from models import LSTM
from sti_config import config


class Trainer:
    def __init__(self):
        # Set Seed
        seed = config.getint('TORCH', 'manual_seed')
        torch.manual_seed(seed)

        # Set Attributes
        self.dataloader = DataHandler()
        self.model = self.get_model(method='lstm')
        self.loss_function = self.get_loss_function(method='cross_entropy')
        self.optimizer = self.get_optimizer(method='adam')
        self.softmax = nn.Softmax(dim=1)
        self.epoch = config.getint('TRAINING', 'epoch')
        self.early_stop_freq = config.getint('TRAINING', 'early_stop_freq')

    def train(self, train_x, train_y, validate_x, validate_y):
        logger.info('Training model')

        # Preprocess data to obtain train_set_dataloader and validation set data (Type: torch)
        train_set_dataloader, validate_x, validate_y = self.dataloader.preprocess_data(train_x, train_y,
                                                                                       validate_x, validate_y)

        # Variables for early stop mechanism
        lowest_val_loss, lowest_val_loss_epoch, early_stop_counter = 9999, 0, 0
        lowest_val_loss_model = pickle.loads(pickle.dumps(self.model.state_dict()))

        for e in range(self.epoch):
            list_of_train_loss = list()  # Logging train loss
            for idx, (sampled_x, sampled_y) in enumerate(train_set_dataloader):
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
                lowest_val_loss_model = pickle.loads(pickle.dumps(self.model.state_dict()))
                early_stop_counter = 0

            if early_stop_counter == self.early_stop_freq:
                self.model.load_state_dict(lowest_val_loss_model)
                logger.info(f'Early stop since validation loss had not drop for {early_stop_freq} epochs')
                logger.info(f'Lowest validation loss {lowest_val_loss} at Epoch: {lowest_val_loss_epoch}')
                break

    def inference(self, train_x):
        train_x = torch.FloatTensor(train_x)
        y_pred = self.model(train_x)
        predicted_class = np.argmax(y_pred.detach().numpy(), axis=1)
        probability_per_class = self.softmax(y_pred).detach().numpy()
        return predicted_class, probability_per_class

    @staticmethod
    def get_model(method='lstm'):
        if method == 'lstm':
            mod = LSTM()
        else:
            logger.info(f"The specified model '{method}' not found. Revert to use 'lstm'")
            mod = LSTM()
            method = 'lstm'

        logger.info(f"Model Used: {method}")
        return mod

    @staticmethod
    def get_loss_function(method='cross_entropy'):
        if method == 'cross_entropy':
            loss = nn.CrossEntropyLoss()
        else:
            logger.info(f"The specified loss function '{method}' not found. Revert to use 'cross_entropy'")
            loss = nn.CrossEntropyLoss()
            method = 'cross_entropy'

        logger.info(f"Loss function Used: {method}")
        return loss

    def get_optimizer(self, method='adam'):
        model_params = self.model.parameters()
        learning_rate = 0.05
        if method == 'adam':
            optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        else:
            logger.info(f"The specified optimizer '{method}' not found. Revert to use 'adam'")
            optimizer = torch.optim.Adam(model_params, lr=learning_rate)
            method = 'adam'

        logger.info(f"Optimizer Used: {method}")
        return optimizer
