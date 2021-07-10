import torch
import torch.nn as nn
from models import LSTM
from logzero import logger
import numpy as np
from datahandler import DataHandler


class Trainer:
    def __init__(self):
        # Set Seed
        torch.manual_seed(123)

        # Set Attributes
        self.model = self.get_model(method='lstm')
        self.loss_function = self.get_loss_function(method='cross_entropy')
        self.optimizer = self.get_optimizer(method='adam')
        self.softmax = nn.Softmax(dim=1)

    def train(self, train_x, train_y):
        # TODO: Put this config into config file later
        epoch = 100

        # Conduct train-test-split, data augmentation and prepare dataloader
        train_x, validate_x, train_y, validate_y = DataHandler.train_test_split(train_x, train_y)
        train_x, train_y = DataHandler.data_augmentation(train_x, train_y)
        train_dataloader = DataHandler.get_data_loader(train_x, train_y, is_train=True)
        validate_x = torch.FloatTensor(validate_x)
        validate_y = torch.LongTensor(validate_y)

        for e in range(epoch):
            total_train_loss = list()
            for idx, (sampled_x, sampled_y) in enumerate(train_dataloader):
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
                total_train_loss.append(train_loss.detach().numpy().mean())

            # Summarize the Train & Validation Loss
            with torch.no_grad():
                y_pred = self.model(validate_x)
                validate_loss = self.loss_function(y_pred, validate_y)
                logger.info(f'epoch: {e}, train loss: {np.mean(total_train_loss)}, validate loss: {validate_loss.mean()}')

    def inference(self, train_x):
        train_x = torch.FloatTensor(train_x)
        y_pred = self.model(train_x)
        predicted_class = np.argmax(y_pred.detach().numpy(), axis=1)
        probability_per_class = self.softmax(y_pred)
        return predicted_class, probability_per_class

    @staticmethod
    def get_model(method='lstm'):
        input_size = 41
        hidden_layer_size = 8
        output_size = 31
        if method == 'lstm':
            mod = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size)
        else:
            logger.info(f"The specified model '{method}' not found. Revert to use 'lstm'")
            mod = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size)
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
        learning_rate = 0.001
        if method == 'adam':
            optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        else:
            logger.info(f"The specified optimizer '{method}' not found. Revert to use 'adam'")
            optimizer = torch.optim.Adam(model_params, lr=learning_rate)
            method = 'adam'

        logger.info(f"Optimizer Used: {method}")
        return optimizer
