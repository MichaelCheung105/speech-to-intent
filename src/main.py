import numpy as np
from logzero import logger
from config import HOME_PATH, config
from trainer import Trainer
from sklearn.metrics import confusion_matrix


class Runner:
    def __init__(self):
        self.trainer = Trainer()

    @staticmethod
    def read_data(data_source):
        data_path = config['DEFAULT']['data_dir']
        file_name = 'data_for_training' if data_source == 'train' else 'test_data'
        data = np.load(f"{HOME_PATH}/{data_path}/{file_name}.npz")
        return data['f0'], data['f1']

    def measure_accuracy(self, labels, predicted_class, y_pred_prob):
        pass

    def run(self):
        experiment_mode = config['DEFAULT']['experiment_mode']
        logger.info(f'experiment mode: {experiment_mode}')

        if 'train' in experiment_mode:
            logger.info('loading train data')
            train_x, train_y = self.read_data(data_source='train')
            self.trainer.train(train_x, train_y)
            predicted_class, probability_per_class = self.trainer.inference(train_x)
            self.measure_accuracy(labels=train_y, predicted_class=predicted_class, y_pred_prob=probability_per_class)

        if 'test' in experiment_mode:
            logger.info('loading test data')
            test_x, test_y = self.read_data(data_source='train')
            predicted_class, probability_per_class = self.trainer.inference(test_x)
            self.measure_accuracy(labels=test_y, predicted_class=predicted_class, y_pred_prob=probability_per_class)


if __name__ == '__main__':
    logger.info('program begins')
    runner = Runner()
    runner.run()
    logger.info('program ends')
