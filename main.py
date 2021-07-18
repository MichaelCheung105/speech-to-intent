import pickle
from datetime import datetime

from logzero import logger

from data_exporter import DataExporter
from data_handler import DataHandler
from metrichandler import MetricHandler
from sti_config import HOME_PATH, config
from trainer import Trainer


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
        model_id = config.get('DEFAULT', 'model_id') if self.experiment_mode == 'evaluate' else self.create_dt
        load_path = f"{HOME_PATH}/{model_path}/{model_id}_model.pkl"
        logger.info(f'Loading model from {load_path}')

        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def inference_and_evaluate(self, result_dict, data_x, data_y, data_i, dataset):
        logger.info(f'Inference and evaluation on {dataset} set')

        predicted_class, probability_per_class = self.trainer.inference(data_x)
        conf_matrix, metrics_df = self.metric_handler.calculate_metrics(labels=data_y, predicted_class=predicted_class)
        predictions_df = self.metric_handler.get_prediction_df(dataset, data_i, data_y,
                                                               probability_per_class, predicted_class)

        # Store evaluation metrics
        result_dict['confusion_matrices'][dataset] = conf_matrix
        result_dict['merged_metrics_df'][dataset] = metrics_df.assign(dataset=dataset)
        result_dict['predictions_df'][dataset] = predictions_df

    def run(self):
        logger.info(f'Experiment Mode: {self.experiment_mode}')

        # Read data, label and respective index
        train_x, train_y, train_i = DataHandler.read_data(data_source='train')
        test_x, test_y, test_i = DataHandler.read_data(data_source='test')

        # Prepare train & validate set
        train_x, val_x, train_y, val_y, train_i, val_i = DataHandler.train_test_split(train_x, train_y, train_i)

        # Prepare result dict to log results
        result_dict = {'learning_curve': [], 'confusion_matrices': {}, 'merged_metrics_df': {}, 'predictions_df': {}}

        # Train or Load model from directory
        if 'train' in self.experiment_mode:
            self.trainer.train(train_x, train_y, val_x, val_y, result_dict)
            self.save_model()
        else:
            self.trainer = self.load_model()

        # Inference and log result
        self.inference_and_evaluate(result_dict, train_x, train_y, train_i, dataset='train')
        self.inference_and_evaluate(result_dict, val_x, val_y, val_i, dataset='validate')
        self.inference_and_evaluate(result_dict, test_x, test_y, test_i, dataset='test')
        self.data_exporter.log_result(result_dict, verbose=True)


if __name__ == '__main__':
    create_dt = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    create_dt = '20210717000000'
    logger.info(f'Experiment {create_dt} begins')
    runner = Runner(create_dt)
    runner.run()
    logger.info(f'Experiment {create_dt} ends')
