from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, CosineSimilarity
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

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

def create_callbacks(callbacks_list, model_name, img_model_name, optimizer_name, learning_rate, min_delta=0.0001, patience=10):
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
        f"./runs/models/{model_name}/{img_model_name}/{optimizer_name}/lr_{learning_rate}",
        monitor='val_loss',
        save_best_only=True,
        mode='min'
        ),
        "tensorboard": TensorBoard(
        log_dir=f'./runs/logs/{model_name}/{img_model_name}/{optimizer_name}/lr_{learning_rate}',
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