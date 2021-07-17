from sti_config import HOME_PATH, config
import seaborn as sns
import matplotlib.pyplot as plt
from logzero import logger
import pandas as pd


class DataExporter:
    def __init__(self, create_dt, experiment_mode):
        self.experiment_mode = experiment_mode
        self.create_dt = create_dt

    def log_result(self, result_dict, verbose=False):
        logger.info(f'Start logging results')

        self.log_evaluation_metrics(result_dict, verbose)
        self.log_confusion_matrix(result_dict)
        self.log_predictions(result_dict)
        if self.experiment_mode == 'train':
            self.log_learning_curve(result_dict)

    def log_predictions(self, result_dict):
        logger.info(f'Step 3: exporting predictions')

        predictions_df = pd.concat(result_dict['predictions_df'].values())
        model_id = config.get('DEFAULT', 'model_id') if self.experiment_mode == 'test' else self.create_dt
        predictions_df = predictions_df.assign(create_dt=self.create_dt,
                                               model_id=model_id,
                                               remarks=config.get('DEFAULT', 'remarks'))
        predictions_df['label'] = predictions_df['label'] + 1  # +1 to make the classes start from 1
        predictions_df['prediced_class'] = predictions_df['prediced_class'] + 1  # +1 to make the classes start from 1

        # Log evaluation metrics as csv
        predictions_path = config.get('DEFAULT', 'predictions_path')
        save_path = f"{HOME_PATH}/{predictions_path}/{self.create_dt}_predictions_df.csv"
        logger.info(f'Saving predictions_df to {save_path}')
        predictions_df.to_csv(save_path, index=False)

    def log_confusion_matrix(self, result_dict):
        logger.info(f'Step 2: exporting confusion matrix')

        # Define plot configs
        num_of_subplots = len(result_dict['confusion_matrices'])
        tick_labels = [str(i + 1) for i in range(31)]

        # Plot confusion matrix
        fig, axes = plt.subplots(nrows=num_of_subplots, figsize=(20, 10 * 3))
        for idx, keys in enumerate(result_dict['confusion_matrices']):
            sns.heatmap(data=result_dict['confusion_matrices'][keys], ax=axes[idx]
                        , xticklabels=tick_labels, yticklabels=tick_labels
                        , cbar=True, cmap='YlGnBu', linewidths=.1, linecolor='b', annot=True, fmt='d')
            axes[idx].set_title(f'Confusion Matrix: {keys} set')
            axes[idx].set_xlabel('Prediction')
            axes[idx].set_ylabel('True Label')

        # Save figure
        confusion_matrix_path = config.get('DEFAULT', 'confusion_matrix_path')
        save_path = f"{HOME_PATH}/{confusion_matrix_path}/{self.create_dt}_confusion_matrix.png"
        logger.info(f'Saving confusion matrix to {save_path}')
        fig.savefig(save_path, bbox_inches='tight')

    def log_evaluation_metrics(self, result_dict, verbose):
        logger.info(f'Step 0: calculating evaluation metrics')

        # Concat the metrics df
        cols = ['category', 'f1_score', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn', 'dataset']
        merged_metrics_df = pd.concat(result_dict['merged_metrics_df'].values())
        merged_metrics_df.columns = cols
        model_id = config.get('DEFAULT', 'model_id') if self.experiment_mode == 'test' else self.create_dt
        merged_metrics_df = merged_metrics_df.assign(create_dt=self.create_dt,
                                                     model_id=model_id,
                                                     remarks=config.get('DEFAULT', 'remarks'))

        # Log evaluation metrics as csv
        logger.info(f'Step 1: exporting evaluation metrics')
        accuracy_path = config.get('DEFAULT', 'accuracy_path')
        save_path = f"{HOME_PATH}/{accuracy_path}/{self.create_dt}_merged_metrics_df.csv"
        logger.info(f'Saving evaluation metrics to {save_path}')
        merged_metrics_df.to_csv(save_path, index=False)

        # Print result
        if verbose:
            pd.set_option('display.max_columns', 20)
            pd.set_option('display.width', 200)
            logger.info(f"Result of Train Set:\n{merged_metrics_df[merged_metrics_df.dataset == 'train']}")
            logger.info(f"Result of Validate Set:\n{merged_metrics_df[merged_metrics_df.dataset == 'validate']}")
            logger.info(f"Result of Test Set:\n{merged_metrics_df[merged_metrics_df.dataset == 'test']}")

    def log_learning_curve(self, result_dict):
        logger.info(f'Step 4: exporting learning curves')
        df = pd.DataFrame(result_dict['learning_curve'], columns=['epoch', 'avg_train_loss', 'avg_validation_loss'])
        learning_curve_path = config.get('DEFAULT', 'learning_curve_path')
        save_path = f"{HOME_PATH}/{learning_curve_path}/{self.create_dt}_learning_curve_df.csv"
        logger.info(f'Saving learning curve to {save_path}')
        df.to_csv(save_path, index=True)
