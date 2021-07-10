import numpy as np
from logzero import logger
from config import HOME_PATH, config


class Runner():
    def __init__(self):
        pass

    @staticmethod
    def read_data(data_source):
        data_path = config['DEFAULT']['data_dir']
        data = np.load(f"{HOME_PATH}/{data_path}/{data_source}_data.npz")
        return data['f0'], data['f1']

    def run(self):
        experiment_mode = config['DEFAULT']['experiment_mode']
        logger.info(f'experiment mode: {experiment_mode}')

        if 'train' in experiment_mode:
            logger.info('loading train data')
            train_x, train_y = self.read_data(data_source='train')

        if 'test' in experiment_mode:
            logger.info('loading test data')
            test_x, test_y = self.read_data(data_source='train')


if __name__ == '__main__':
    logger.info('program begins')
    runner = Runner()
    runner.run()
    logger.info('program ends')
