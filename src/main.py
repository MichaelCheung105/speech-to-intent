import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from logzero import logger

from datahandler import DataHandler
from metrichandler import MetricHandler
from sti_config import HOME_PATH, config
from trainer import Trainer


class Runner:
    def __init__(self, create_dt):
        self.create_dt = create_dt
        self.trainer = Trainer()
        self.metric_handler = MetricHandler()
        self.experiment_mode = config.get('DEFAULT', 'experiment_mode')

    @staticmethod
    def read_data(data_source):
        logger.info(f'Loading {data_source} data')

        data_path = config.get('DEFAULT', 'data_dir')
        file_name = 'data_for_training' if data_source == 'train' else 'test_data'
        data = np.load(f"{HOME_PATH}/{data_path}/{file_name}.npz")
        samples = data['f0']
        labels = data['f1'] - 1  # -1 to make the classes starts with 0
        return samples, labels

    def log_result(self, result_dict, verbose=False):
        logger.info(f'Logging Result')

        cols = ['category', 'logloss', 'accuracy', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn', 'dataset']
        merged_metrics_df = pd.concat(result_dict['merged_metrics_df'])
        merged_metrics_df.columns = cols
        merged_metrics_df = merged_metrics_df.assign(experiment_id=self.create_dt,
                                                     remarks=config.get('DEFAULT', 'remarks'))

        # Print result
        if verbose:
            pd.set_option('display.max_columns', 20)
            pd.set_option('display.width', 200)
            logger.info(f"\n{merged_metrics_df[merged_metrics_df.dataset == 'train']}")
            logger.info(f"\n{merged_metrics_df[merged_metrics_df.dataset == 'test']}")

        # Log result as csv
        accuracy_path = config.get('DEFAULT', 'accuracy_path')
        save_path = f"{HOME_PATH}/{accuracy_path}/{self.create_dt}_merged_metrics_df.csv"
        logger.info(f'Saving evaluation metrics to {save_path}')
        merged_metrics_df.to_csv(save_path)

    def save_model(self):
        model_path = config.get('DEFAULT', 'model_path')
        save_path = f"{HOME_PATH}/{model_path}/{self.create_dt}_model.pkl"
        logger.info(f'Saving model to {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.trainer.model, f)

    def load_model(self):
        model_path = config.get('DEFAULT', 'model_path')
        model_id = config.get('DEFAULT', 'model_id') if self.experiment_mode == 'test' else self.create_dt
        load_path = f"{HOME_PATH}/{model_path}/{model_id}_model.pkl"
        logger.info(f'Loading model from {load_path}')

        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def inference_and_evaluate(self, result_dict, data_x, data_y, dataset):
        logger.info(f'Inference and evaluation on {dataset} set')

        predicted_class, probability_per_class = self.trainer.inference(data_x)
        conf_matrix, metrics_df = self.metric_handler.calculate_metrics(labels=data_y,
                                                                        predicted_class=predicted_class,
                                                                        y_pred_prob=probability_per_class)
        # Store evaluation metrics
        result_dict['confusion_matrices'][dataset] = conf_matrix
        result_dict['merged_metrics_df'].append(metrics_df.assign(dataset=dataset))

    def run(self):
        logger.info(f'Experiment Mode: {self.experiment_mode}')

        # Prepare a dictionary to log training results
        result_dict = {'confusion_matrices': {'train': None, 'validate': None, 'test': None},
                       'merged_metrics_df': []
                       }

        if 'train' in self.experiment_mode:
            train_x, train_y = self.read_data(data_source='train')
            train_x, validate_x, train_y, validate_y = DataHandler.train_test_split(train_x, train_y)
            self.trainer.train(train_x, train_y, validate_x, validate_y)
            self.inference_and_evaluate(result_dict, train_x, train_y, dataset='train')
            self.inference_and_evaluate(result_dict, validate_x, validate_y, dataset='validate')
            self.save_model()

        if 'test' in self.experiment_mode:
            test_x, test_y = self.read_data(data_source='test')
            self.trainer.model = self.load_model()
            self.inference_and_evaluate(result_dict, test_x, test_y, dataset='Test')

        # Log results
        self.log_result(result_dict, verbose=False)


if __name__ == '__main__':
    create_dt = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    logger.info(f'Experiment {create_dt} begins')
    runner = Runner(create_dt)
    runner.run()
    logger.info(f'Experiment {create_dt} ends')
