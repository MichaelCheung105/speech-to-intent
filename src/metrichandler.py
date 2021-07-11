import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss


class MetricHandler:
    def __init__(self):
        pass

    def calculate_metrics(self, labels, predicted_class, y_pred_prob):
        tp_ls, fp_ls, fn_ls, tn_ls, metric_ls = list(), list(), list(), list(), list()
        conf_matrix = confusion_matrix(labels, predicted_class)

        # Calculate per category metrics
        for cat_idx in range(conf_matrix.shape[0]):  # Loop over all 31 categories
            category = str(cat_idx + 1)  # +1 since the true labels start at 1 instead of 0
            logloss, accuracy, precision, recall, tp, fp, fn, tn = self.get_per_category_metrics(conf_matrix,
                                                                                                 cat_idx,
                                                                                                 labels,
                                                                                                 y_pred_prob
                                                                                                 )
            tp_ls.append(tp)
            fp_ls.append(fp)
            fn_ls.append(fn)
            tn_ls.append(tn)
            metric_ls.append([category, logloss, accuracy, precision, recall, tp, fp, fn, tn])

        # Calculate allover metric
        logloss, accuracy, precision, recall, tp, fp, fn, tn = self.get_overall_metrics(conf_matrix,
                                                                                        tp_ls, fp_ls, fn_ls, tn_ls,
                                                                                        labels, y_pred_prob)
        metric_ls.append(['overall', logloss, accuracy, precision, recall, tp, fp, fn, tn])

        # Return the metrics as dataframe
        metrics_df = pd.DataFrame(metric_ls[::1])
        return conf_matrix, metrics_df

    @staticmethod
    def get_per_category_metrics(conf_matrix, cat_idx, labels, y_pred_prob):
        logloss = log_loss(labels == cat_idx, y_pred_prob[:, cat_idx])
        num_all_label = conf_matrix.sum()
        num_pred_label = conf_matrix[:, cat_idx].sum()
        num_real_label = conf_matrix[cat_idx, :].sum()
        tp = conf_matrix[cat_idx, cat_idx]
        fp = num_pred_label - tp
        fn = num_real_label - tp
        tn = num_all_label - tp - fp - fn
        precision = tp / num_pred_label if num_pred_label != 0 else 0
        recall = tp / num_real_label if num_real_label != 0 else 0
        accuracy = tp / (tp + fp + fn + tn)
        return logloss, accuracy, precision, recall, tp, fp, fn, tn

    @ staticmethod
    def get_overall_metrics(conf_matrix, tp_ls, fp_ls, fn_ls, tn_ls, labels, y_pred_prob):
        tp = sum(tp_ls)
        fp = sum(fp_ls)
        fn = sum(fn_ls)
        tn = sum(tn_ls)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = tp / conf_matrix.sum()
        logloss = log_loss(labels, y_pred_prob)
        return logloss, accuracy, precision, recall, tp, fp, fn, tn
