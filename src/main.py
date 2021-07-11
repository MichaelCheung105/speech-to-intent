from datetime import datetime

import numpy as np
import pandas as pd
from logzero import logger

from metrichandler import MetricHandler
from sti_config import HOME_PATH, config
from trainer import Trainer


class Runner:
    def __init__(self, create_dt):
        self.create_dt = create_dt
        self.trainer = Trainer()
        self.metric_handler = MetricHandler()

    @staticmethod
    def read_data(data_source):
        data_path = config.get('DEFAULT', 'data_dir')
        file_name = 'data_for_training' if data_source == 'train' else 'test_data'
        data = np.load(f"{HOME_PATH}/{data_path}/{file_name}.npz")
        samples = data['f0']
        labels = data['f1'] - 1  # -1 to make the classes starts with 0
        return samples, labels

    def log_result(self, result_dict):
        cols = ['category', 'logloss', 'accuracy', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn', 'dataset', 'remarks']
        merged_metrics_df = pd.concat(result_dict['merged_metrics_df']).assign(remarks=config.get('DEFAULT', 'remarks'))
        merged_metrics_df.columns = cols
        merged_metrics_df.assign(experiment_id=self.create_dt)
        # TODO: log result as csv
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 200)
        logger.info(f"\n{merged_metrics_df[merged_metrics_df.dataset == 'train']}")
        logger.info(f"\n{merged_metrics_df[merged_metrics_df.dataset == 'test']}")

    def run(self):
        experiment_mode = config.get('DEFAULT', 'experiment_mode')
        logger.info(f'experiment mode: {experiment_mode}')

        # Prepare a dictionary to log training results
        result_dict = {'confusion_matrices': [],
                       'merged_metrics_df': []
                       }

        if 'train' in experiment_mode:
            logger.info('loading train data')
            train_x, train_y = self.read_data(data_source='train')
            # self.trainer.train(train_x, train_y)
            predicted_class, probability_per_class = self.trainer.inference(train_x)
            train_conf_matrix, train_metrics_df = self.metric_handler.calculate_metrics(labels=train_y,
                                                                                        predicted_class=predicted_class,
                                                                                        y_pred_prob=probability_per_class)
            result_dict['merged_metrics_df'].append(train_metrics_df.assign(dataset='train'))

        if 'test' in experiment_mode:
            logger.info('loading test data')
            test_x, test_y = self.read_data(data_source='train')
            predicted_class, probability_per_class = self.trainer.inference(test_x)
            test_conf_matrix, test_metrics_df = self.metric_handler.calculate_metrics(labels=test_y,
                                                                                      predicted_class=predicted_class,
                                                                                      y_pred_prob=probability_per_class)
            result_dict['merged_metrics_df'].append(test_metrics_df.assign(dataset='test'))

        # Log result
        self.log_result(result_dict)


if __name__ == '__main__':
    create_dt = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    logger.info(f'Experiment {create_dt} begins')
    runner = Runner(create_dt)
    runner.run()
    logger.info(f'Experiment {create_dt} ends')
