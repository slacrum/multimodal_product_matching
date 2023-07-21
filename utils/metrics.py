import numpy as np
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, CosineSimilarity, binary_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.losses import cosine_similarity
from tensorflow import reduce_mean
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from utils.img_processing import load_img_model


def create_metrics(metric_list):
    metrics_dict = {
        "recall": Recall(thresholds=0.5, top_k=1, class_id=None, name="recall", dtype=None),
        "precision": Precision(thresholds=0.5, top_k=1, class_id=None, name="precision", dtype=None),
        "binary_accuracy": BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
        "cosine_similarity": CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
    }

    return [metrics_dict[metric]
            for metric in metric_list
            if metric in metrics_dict]


def create_callbacks(
        callbacks_list, log_dir, model_name, img_model_name, optimizer_name,
        learning_rate, cls, min_delta=0.0001, patience=3):
    callbacks_dict = {
        "early_stopping": EarlyStopping(
            monitor='val_loss',
            min_delta=min_delta,
            patience=patience,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=False
        ),
        "model_checkpoint": ModelCheckpoint(
            f"{log_dir}/models/{model_name}/cls_{cls}/{img_model_name}/{optimizer_name}/lr_{learning_rate}",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        "tensorboard": TensorBoard(
            log_dir=f'{log_dir}/logs/{model_name}/cls_{cls}/{img_model_name}/{optimizer_name}/lr_{learning_rate}',
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
    }

    return [callbacks_dict[callback]
            for callback in callbacks_list
            if callback in callbacks_dict]


def plot_metrics(
        history, metrics, model_name, img_model_name, optimizer_name,
        learning_rate, cls):
    metrics = ["loss"] + metrics
    for metric in metrics:
        plt.plot(history[metric])
        plt.plot(history[f'val_{metric}'])
        plt.title(
            f'{metric} | {model_name} | {img_model_name} | lr: {learning_rate} ({optimizer_name}) | cls > {cls}')
        plt.ylabel(f'{metric}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


def evaluate(
        model, x, labels_test, log_dir, model_name, img_model_name,
        optimizer_name, learning_rate, cls, batch_size=1, triplet_model=False):
    model.evaluate(x, labels_test, batch_size=batch_size)

    results = model.predict(x)

    if triplet_model:
        img_predictions = results[:, :results.shape[1]//2]
        text_predictions = results[:, results.shape[1]//2:results.shape[1]]
        results = cosine_similarity(
            img_predictions, text_predictions
        ).numpy().reshape(-1, 1)
        results = -results

    total_params = load_img_model(img_model_name).count_params() + model.count_params()

    np.save(
        f'{log_dir}/logs/{model_name}/cls_{cls}/{img_model_name}/{optimizer_name}/lr_{learning_rate}/metrics',
        np.array(
            [roc_curve(labels_test, results),
             precision_recall_curve(labels_test, results),
             labels_test, results,
             model.count_params(),
             total_params],
            dtype=object))


def extract_metrics_config(config):
    return {
        "log_dir": config["model"]["training"]["log_dir"],
        "model_name": config["model"]["name"],
        "img_model_name": config["img_model"],
        "optimizer_name": config["model"]["training"]["optimizer"],
        "learning_rate": config["model"]["training"]["learning_rate"],
        "batch_size": config["model"]["training"]["batch_size"],
        "epochs": config["model"]["training"]["epochs"],
        "cls": config["data"]["cls"]
    }


class Metric(object):
    def __init__(
            self, log_dir, model_name, img_model_name, optimizer_name,
            learning_rate, batch_size, epochs, cls):
        self.log_dir = log_dir
        self.model_name = model_name
        self.img_model_name = img_model_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.cls = cls
        self._load_metric()

    def _load_metric(self):
        metrics_path = f'{self.log_dir}/logs/{self.model_name}/cls_{self.cls}/{self.img_model_name}/{self.optimizer_name}/lr_{self.learning_rate}/metrics.npy'
        metrics = np.load(metrics_path, allow_pickle=True)
        self.metrics = metrics

    def plot_roc(self):
        roc = plt.plot(
            self.metrics[0][0],
            self.metrics[0][1],
            label="%s | %s | lr: %s (%s) | cls > %s (AUC = %0.3f)" %
            (self.model_name, self.img_model_name, self.learning_rate, self.
             optimizer_name, self.cls,
             auc(self.metrics[0][0],
                 self.metrics[0][1])))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        return roc

    def plot_prc(self):
        prc = plt.plot(
            self.metrics[1][1],
            self.metrics[1][0],
            label="%s | %s | lr: %s (%s) | cls > %s (AP = %0.3f)" %
            (self.model_name, self.img_model_name, self.learning_rate, self.
             optimizer_name, self.cls,
             auc(self.metrics[1][1],
                 self.metrics[1][0])))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        return prc

    def _gmean(self):
        fpr, tpr = self.metrics[0][0], self.metrics[0][1]
        return np.sqrt(tpr * (1-fpr))

    def _J(self):
        fpr, tpr = self.metrics[0][0], self.metrics[0][1]
        return tpr - fpr

    def _fscore(self):
        precision, recall = self.metrics[1][0], self.metrics[1][1]
        numerator = (2 * precision * recall)
        denom = (precision + recall)
        # sometimes precision and recall is 0, therefore denom is 0, so we get NaN values. np.divide() can prevent that
        fscores = np.divide(numerator, denom, out=np.zeros_like(
            denom), where=(denom != 0))
        return fscores

    def _accuracy(self, mode):
        roc, prc, y_true, y_pred, *_ = self.metrics
        if mode == "gmean":
            metric = self._gmean()
            threshold = roc[2][np.argmax(metric)]
        elif mode == "J":
            metric = self._J()
            threshold = roc[2][np.argmax(metric)]
        elif mode == "fscore":
            metric = self._fscore()
            threshold = prc[2][np.argmax(metric)]
        return metric.max(), threshold, reduce_mean(
            binary_accuracy(
                y_true[["label"]], y_pred, threshold=threshold
            )
        ).numpy()

    def optimize_threshold(self):
        gmean, threshold_gmean, acc_gmean = self._accuracy("gmean")
        J, threshold_J, acc_J = self._accuracy("J")
        precision, recall = self.metrics[1][0][np.argmax(
            self._fscore())], self.metrics[1][1][np.argmax(self._fscore())]
        fscore_best, threshold_fscore, acc_fscore = self._accuracy("fscore")
        model_params = self.metrics[4]
        total_params = self.metrics[5]
        return {'Model name': self.model_name,
                'Image CNN': self.img_model_name,
                '# Parameters (without Image CNN)': model_params,
                'Total parameters': total_params,
                'Optimizer': self.optimizer_name,
                'lr': self.learning_rate,
                'Batch size': self.batch_size,
                '# Epochs': self.epochs,
                'cls': self.cls,
                'AUC-ROC': auc(self.metrics[0][0], self.metrics[0][1]),
                'AUC-PRC': auc(self.metrics[1][1], self.metrics[1][0]),
                'G-Mean': gmean,
                'Threshold G-Mean': threshold_gmean,
                'Accuracy G-Mean': acc_gmean,
                'J': J,
                'Threshold J': threshold_J,
                'Accuracy J': acc_J,
                'Precision': precision,
                'Recall': recall,
                'F-Score': fscore_best,
                'Threshold F-Score': threshold_fscore,
                'Accuracy F-Score': acc_fscore
                }
