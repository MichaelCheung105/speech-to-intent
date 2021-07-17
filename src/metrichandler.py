import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class MetricHandler:
    def __init__(self):
        pass

    def calculate_metrics(self, labels, predicted_class):
        tp_ls, fp_ls, fn_ls, tn_ls, metric_ls = list(), list(), list(), list(), list()
        conf_matrix = confusion_matrix(labels, predicted_class)

        # Calculate per category metrics
        for cat_idx in range(conf_matrix.shape[0]):  # Loop over all 31 categories
            category = str(cat_idx + 1)  # +1 since the true labels start at 1 instead of 0
            f1_score, precision, recall, tp, fp, fn, tn = self.get_per_category_metrics(conf_matrix, cat_idx)
            tp_ls.append(tp)
            fp_ls.append(fp)
            fn_ls.append(fn)
            tn_ls.append(tn)
            metric_ls.append([category, f1_score, precision, recall, tp, fp, fn, tn])

        # Calculate allover metric
        f1_score, precision, recall, tp, fp, fn, tn = self.get_micro_average(tp_ls, fp_ls, fn_ls, tn_ls)
        metric_ls.append(['micro-average', f1_score, precision, recall, tp, fp, fn, tn])

        # Return the metrics as dataframe
        metrics_df = pd.DataFrame(metric_ls)
        return conf_matrix, metrics_df

    @staticmethod
    def get_per_category_metrics(conf_matrix, cat_idx):
        num_all_label = conf_matrix.sum()
        num_pred_label = conf_matrix[:, cat_idx].sum()
        num_real_label = conf_matrix[cat_idx, :].sum()
        tp = conf_matrix[cat_idx, cat_idx]
        fp = num_pred_label - tp
        fn = num_real_label - tp
        tn = num_all_label - tp - fp - fn
        precision = tp / num_pred_label if num_pred_label != 0 else 0
        recall = tp / num_real_label if num_real_label != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        return f1_score, precision, recall, tp, fp, fn, tn

    @ staticmethod
    def get_micro_average(tp_ls, fp_ls, fn_ls, tn_ls):
        tp = sum(tp_ls)
        fp = sum(fp_ls)
        fn = sum(fn_ls)
        tn = sum(tn_ls)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall, tp, fp, fn, tn

    def get_prediction_df(self, dataset, data_i, data_y, probability_per_class, predicted_class):
        predicted_class_df = pd.DataFrame(data={'dataset': dataset, 'indices': data_i, 'label': data_y,
                                                'predicted_class': predicted_class,
                                                'predicted_class_probability': np.max(probability_per_class, 1)})

        other_class_df = pd.DataFrame(data=probability_per_class, columns=range(1, probability_per_class.shape[1]+1))

        return pd.concat([predicted_class_df, other_class_df], axis=1)