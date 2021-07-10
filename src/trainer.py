import torch
import torch.nn as nn
from models import LSTM
from logzero import logger
import numpy as np


class Trainer:
    def __init__(self):
        # Set Seed
        torch.manual_seed(123)

        # Set Attributes
        self.model = self.get_model(method='lstm')
        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.softmax = nn.Softmax(dim=1)

    def train(self, train_x, train_y):
        epoch = 3
        train_y = torch.LongTensor(train_y - 1)  # TODO: Handle this train_y -1 more beautifully
        for i in range(epoch):
            # Sample Data
            # TODO: Use DataLoader
            total_rows = train_x.shape[0]
            train_num = int(total_rows * 0.8)
            train_x, validate_x = train_x[:train_num], train_x[train_num:]
            train_y, validate_y = train_y[:train_num], train_y[train_num:]

            # Reset gradient
            self.optimizer.zero_grad()

            # Get prediction
            y_pred, _, _ = self.inference(train_x, verbose=False)

            # Calculate loss
            train_loss = self.loss_function(y_pred, train_y)

            # Backpropagation
            train_loss.backward()
            self.optimizer.step()

            # Log Loss
            with torch.no_grad():
                y_pred, _, _ = self.inference(validate_x, verbose=False)
                validate_loss = self.loss_function(y_pred, validate_y)
                logger.info(f'train loss: {train_loss.sum()}, validate loss: {validate_loss.sum()}')

    def inference(self, train_x, verbose):
        train_x = torch.FloatTensor(train_x)
        y_pred = self.model(train_x)
        if verbose:
            predicted_class = np.argmax(y_pred.detach().numpy(), axis=1)
            probability_per_class = self.softmax(y_pred)
        else:
            predicted_class = None
            probability_per_class = None
        return y_pred, predicted_class, probability_per_class

    @staticmethod
    def get_model(method='lstm'):
        input_size = 41
        hidden_layer_size = 8
        output_size = 31
        if method == 'some method':
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
