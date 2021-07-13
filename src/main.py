import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from logzero import logger

from data_handler import DataHandler
from metrichandler import MetricHandler
from sti_config import HOME_PATH, config
from trainer import Trainer
from data_exporter import DataExporter


class Runner:
    def __init__(self, create_dt):
        self.create_dt = create_dt
        self.experiment_mode = config.get('DEFAULT', 'experiment_mode')
        self.trainer = Trainer()
        self.metric_handler = MetricHandler()
        self.data_exporter = DataExporter(self.create_dt, self.experiment_mode)

    def save_model(self):
        model_path = config.get('DEFAULT', 'model_path')
        save_path = f"{HOME_PATH}/{model_path}/{self.create_dt}_model.pkl"
        logger.info(f'Saving model to {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.trainer, f)

    def load_model(self):
        model_path = config.get('DEFAULT', 'model_path')
        model_id = config.get('DEFAULT', 'model_id') if self.experiment_mode == 'test' else self.create_dt
        load_path = f"{HOME_PATH}/{model_path}/{model_id}_model.pkl"
        logger.info(f'Loading model from {load_path}')

        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def inference_and_evaluate(self, result_dict, data_x, data_y, data_i, dataset):
        logger.info(f'Inference and evaluation on {dataset} set')

        predicted_class, probability_per_class = self.trainer.inference(data_x)
        conf_matrix, metrics_df = self.metric_handler.calculate_metrics(labels=data_y,
                                                                        predicted_class=predicted_class,
                                                                        y_pred_prob=probability_per_class)
        # Store evaluation metrics
        result_dict['confusion_matrices'][dataset] = conf_matrix
        result_dict['merged_metrics_df'][dataset] = metrics_df.assign(dataset=dataset)
        result_dict['predictions_df'][dataset] = pd.DataFrame(data={'dataset': dataset,
                                                                    'indices': data_i,
                                                                    'label': data_y,
                                                                    'prediced_probability': np.max(probability_per_class, 1),
                                                                    'prediced_class': predicted_class})

    def run(self):
        logger.info(f'Experiment Mode: {self.experiment_mode}')

        # Prepare a dictionary to log training results
        result_dict = {'confusion_matrices': {}, 'merged_metrics_df': {}, 'predictions_df': {}}

        if 'train' in self.experiment_mode:
            train_x, train_y = DataHandler.read_data(data_source='train')
            train_x, train_i = DataHandler.preprocess(train_x)
            train_x, validate_x, train_y, validate_y, train_i, validate_i = DataHandler.train_test_split(train_x,
                                                                                                         train_y,
                                                                                                         train_i
                                                                                                         )
            self.trainer.train(train_x, train_y, validate_x, validate_y)
            self.inference_and_evaluate(result_dict, train_x, train_y, train_i, dataset='train')
            self.inference_and_evaluate(result_dict, validate_x, validate_y, validate_i, dataset='validate')

            if config.getboolean('DEFAULT', 'is_save_result'):
                self.save_model()

        if 'test' in self.experiment_mode:
            self.trainer = self.load_model()
            test_x, test_y = DataHandler.read_data(data_source='test')
            test_x, test_i = DataHandler.preprocess(test_x)
            self.inference_and_evaluate(result_dict, test_x, test_y, test_i, dataset='test')

        # Log results
        if config.getboolean('DEFAULT', 'is_save_result'):
            self.data_exporter.log_result(result_dict, verbose=True)


if __name__ == '__main__':
    create_dt = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    logger.info(f'Experiment {create_dt} begins')
    runner = Runner(create_dt)
    runner.run()
    logger.info(f'Experiment {create_dt} ends')
