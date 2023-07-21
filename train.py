import sys
############### only for bwHPC cluster ###############
sys.path.remove(
    '/home/es/es_es/es_kamait02/.local/lib/python3.9/site-packages')
sys.path.append(
    '/home/es/es_es/es_kamait02/.local/lib/python3.9/site-packages')
######################################################
from utils.metrics import create_metrics, create_callbacks, evaluate, extract_metrics_config, Metric
from models.mnn_btl import MNNBTL
from models.mnn_em import MNNEM, ExtendedMNNEM
from models.char_cnn_zhang import CharCNNZhang
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import class_weight
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import argparse
from data_loader.abo import ABO
from utils.text_processing import CharTokenizer
from utils.img_processing import load_img_model, create_embeddings_from


def handle_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="JSON config")
    parser.add_argument("--save_embeddings", help="Save current embeddings and data for later reuse", action="store_true")
    parser.add_argument("--load_embeddings", help="Load saved embeddings (requires `--save_embeddings` to have been enabled in a previous run)", action="store_true")
    parser.add_argument("--no_train", help="Skip training (and evaluation, by extension)", action="store_true", default=False)
    parser.add_argument("--no_eval", help="Skip evaluation", action="store_true", default=False)
    return parser.parse_args()


def load_dataset(path):
    abo = ABO(path=path,
              download=True,
              extract=True,
              preprocess=True,
              undersample=True,
              alt_augment=True,
              txt_augment=True,
              random_deletion=True,
              export_csv=True).data
    return abo


def load_data_and_embeddings(save_path):
    data = pd.read_csv(f"{save_path}/data.csv")
    data = data.drop({"Unnamed: 0"}, axis=1)

    img = np.load(f"{save_path}/img.npy", allow_pickle=True)
    text = np.load(f"{save_path}/text.npy", allow_pickle=True)
    text2 = np.load(f"{save_path}/text2.npy", allow_pickle=True)
    weights = np.load(f"{save_path}/text_weights.npy", allow_pickle=True)

    return data, img, text, text2, weights


def save_data_and_embeddings(save_path, data, img, text, text2, weights):
    os.makedirs(save_path, exist_ok=True)

    np.save(f"{save_path}/img.npy", img)
    np.save(f"{save_path}/text.npy", text)
    np.save(f"{save_path}/text2.npy", text2)
    np.save(f"{save_path}/text_weights.npy", weights)

    data.to_csv(f"{save_path}/data.csv")


def handle_split(x, y_col, cls, test_split):
    x["product_type_count"] = x.groupby(
        ["product_type"])["product_type"].transform("count")
    x = x[x["product_type_count"] > cls]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        x[[y_col]],
        stratify=x[["product_type"]],
        test_size=test_split,
        random_state=42)

    return x_train, x_test, y_train, y_test


def get_train_test_embeddings(x_train, x_test, img, text, text2):
    return img[x_train.index], img[x_test.index], text[x_train.index], text[x_test.index], text2[x_train.index], text2[x_test.index]


def create_class_weights(y_train):
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=y_train["product_type"].unique(),
        y=y_train["product_type"])

    class_weights_dict = {}
    class_weights_dict_transform = {}
    i = 0

    for pt, cw in zip(y_train["product_type"].unique(), class_weights):
        class_weights_dict[i] = cw
        class_weights_dict_transform[pt] = i
        i += 1
    y_train["product_type"] = y_train["product_type"].apply(
        lambda x: class_weights_dict_transform[x])

    return y_train, class_weights_dict


def build_model(config, weights):
    print("Build Char CNN...")
    char_cnn_config = {
        "input_size": config["data"]["input_size"],
        "embedding_size": weights.shape[1],
        "conv_layers": config["char_cnn_zhang"]["conv_layers"],
        "fc_layers": config["char_cnn_zhang"]["fc_layers"],
        "output_size": config["char_cnn_zhang"]["output_size"],
        "embedding_weights": weights
    }
    char_cnn = CharCNNZhang(**char_cnn_config)

    if config["model"]["type"] == "mnn_em" or config["model"]["type"] == "mnn_btl":
        head_config = {
            "img_input_size": img.shape[1],
            "txt_input_size": config["data"]["input_size"],
            "img_fc_layers": config["model"]["img_fc_layers"],
            "txt_fc_layers": config["model"]["txt_fc_layers"],
            "extended": False,
        }
        if config["model"]["type"] == "mnn_em":
            print("Build MNN-EM...")
            model = MNNEM(
                head_config=head_config, char_cnn=char_cnn.model,
                combined_fc_layers=config["model"]["combined_fc_layers"],
                learning_rate=config["model"]["training"]["learning_rate"],
                metrics=create_metrics(
                    config["model"]["training"]["metrics"]))
        elif config["model"]["type"] == "mnn_btl":
            print("Build MNN-BTL...")
            model = MNNBTL(
                head_config=head_config,
                char_cnn=char_cnn.model,
                learning_rate=config["model"]["training"]["learning_rate"],
                margin=config["model"]["margin"],
                lambda_1=config["model"]["lambda_1"],
                lambda_2=config["model"]["lambda_2"],
                mining=config["model"]["mining"])
    elif config["model"]["type"] == "ext_mnn_em":
        print("Build extended MNN-EM...")
        head_1_config = {
            "img_input_size": img.shape[1],
            "txt_input_size": config["data"]["input_size"],
            "img_fc_layers": config["model"]["img_1_fc_layers"],
            "txt_fc_layers": config["model"]["txt_1_fc_layers"],
            "extended": True,
        }
        head_2_config = {
            "img_input_size": config["model"]["img_1_fc_layers"][-1],
            "txt_input_size": config["data"]["input_size"],
            "img_fc_layers": config["model"]["combined_1_fc_layers"],
            "txt_fc_layers": config["model"]["txt_2_fc_layers"],
            "extended": False,
        }
        model = ExtendedMNNEM(
            head_1_config=head_1_config, head_2_config=head_2_config,
            char_cnn=char_cnn.model,
            combined_fc_layers=config["model"]["combined_2_fc_layers"],
            learning_rate=config["model"]["training"]["learning_rate"],
            metrics=create_metrics(
                config["model"]["training"]["metrics"]))

    return model.model


if __name__ == "__main__":
    args = handle_args()

    config = json.load(open(args.config))

    img_model_name = config["img_model"]

    save_path = os.path.join(config["data"]["path"],
                             f"embeddings/{img_model_name}")

    if args.load_embeddings:
        print("Load saved data and embeddings...")
        data, img, text, text2, weights = load_data_and_embeddings(save_path)
    else:
        print("Create data loader...")
        data = load_dataset(config["data"]["path"])

        print("Create image embeddings...")
        img = data[["path"]]
        img_model = load_img_model(img_model_name)
        img = create_embeddings_from(img_model,
                                     img,
                                     os.path.join(config["data"]["path"],
                                                  "images/small"),
                                     batch_size=2048)

        print("Create text embeddings...")
        tk = CharTokenizer(config["data"]["alphabet"])
        text = tk.tokenize(data["description"]).numpy()
        text2 = tk.tokenize(data["description2"]).numpy()
        weights = tk.create_embedding_weights()

    if args.save_embeddings:
        print("Save data and embeddings...")
        save_data_and_embeddings(save_path, data, img, text, text2, weights)

        print("Verifying import...")
        data, img, text, text2, weights = load_data_and_embeddings(save_path)

    if not args.no_train:
        if config["model"]["type"] == "mnn_btl":
            print("Split data into ground truth and false samples...")
            ground_truth = data[data["label"] == 1]
            false_samples = data[data["label"] == 0]
            data = ground_truth

            img_false = img[false_samples.index]
            text_false = text[false_samples.index]
            text2_false = text2[false_samples.index]

            img = img[data.index]
            text = text[data.index]
            text2 = text2[data.index]

            data = data.reset_index(drop=True)
            false_samples = false_samples.reset_index(drop=True)

            print("Split false samples into train/test...")
            x_train_false, x_test_false, y_train_false, y_test_false = handle_split(
                x=false_samples,
                y_col="product_type" if config["model"]["type"] == "mnn_btl" else "label",
                cls=config["data"]["cls"],
                test_split=config["model"]["training"]["test_split"])

            img_train_false, img_test_false, text_train_false, text_test_false, text2_train_false, text2_test_false = get_train_test_embeddings(
                x_train_false, x_test_false, img_false, text_false, text2_false)

        print("Split data into train/test...")
        x_train, x_test, y_train, y_test = handle_split(
            x=data, y_col="product_type"
            if config["model"]["type"] == "mnn_btl" else "label",
            cls=config["data"]["cls"],
            test_split=config["model"]["training"]["test_split"])

        img_train, img_test, text_train, text_test, text2_train, text2_test = get_train_test_embeddings(
            x_train, x_test, img, text, text2)

        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        n_folds = config["model"]["training"]["n_folds"]

        kfold = KFold(n_splits=n_folds)

        k = 1
        hists = []
        final_accs = []
        model_name = config["model"]["name"]
        for train_idx, val_idx in kfold.split(x_train, y_train):
            print(f"k = {k}/{n_folds}:")
            config["model"]["name"] = model_name + f"_k_{k}"

            x_train_fold = x_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]

            x_val_fold = x_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            img_train_fold, img_val_fold, text_train_fold, text_val_fold, text2_train_fold, text2_val_fold = get_train_test_embeddings(
                x_train_fold, x_val_fold, img_train, text_train, text2_train)

            if config["model"]["type"] == "mnn_btl":
                print("Create class weights...")
                y_train_fold, class_weights_dict = create_class_weights(
                    y_train_fold)

            model = build_model(config, weights)

            print("Create training callbacks...")
            callbacks = create_callbacks(
                callbacks_list=config["model"]["training"]["callbacks"],
                log_dir=config["model"]["training"]["log_dir"],
                model_name=config["model"]["name"],
                img_model_name=img_model_name,
                optimizer_name=config["model"]["training"]["optimizer"],
                learning_rate=config["model"]["training"]["learning_rate"],
                cls=config["data"]["cls"],
                patience=10)

            print("Start training...")
            history = model.fit(
                x=[img_train_fold, text_train_fold, text2_train_fold]
                if config["model"]["type"] == "ext_mnn_em"
                else [img_train_fold, text_train_fold], y=y_train_fold,
                epochs=config["model"]["training"]["epochs"],
                validation_data=([img_val_fold, text_val_fold, text2_val_fold], y_val_fold)
                if config["model"]["type"] == "ext_mnn_em"
                else ([img_val_fold, text_val_fold], y_val_fold),
                batch_size=config["model"]["training"]["batch_size"],
                callbacks=callbacks, class_weight=class_weights_dict
                if config["model"]["type"] == "mnn_btl" else None)
            print("Training complete.")

            hists.append(history)

            if not args.no_eval:
                print("Evaluating model performance...")
                if config["model"]["type"] == "mnn_btl":
                    x_test = pd.concat(
                        [x_test.reset_index(drop=True),
                         x_test_false.reset_index(drop=True)])
                    img_test = np.concatenate([img_test, img_test_false])
                    text_test = np.concatenate([text_test, text_test_false])
                    y_test = x_test[["label"]]

                evaluate(
                    model=model, x=[img_test, text_test, text2_test]
                    if config["model"]["type"] == "ext_mnn_em"
                    else [img_test, text_test], labels_test=y_test,
                    log_dir=config["model"]["training"]["log_dir"],
                    model_name=config["model"]["name"],
                    img_model_name=img_model_name,
                    optimizer_name=config["model"]["training"]["optimizer"],
                    learning_rate=config["model"]["training"]["learning_rate"],
                    cls=config["data"]["cls"],
                    batch_size=config["model"]["training"]["batch_size"],
                    triplet_model=True
                    if config["model"]["type"] == "mnn_btl" else False)

                print("Evaluation complete.")
                metric = Metric(**extract_metrics_config(config))

                metrics_df = pd.DataFrame.from_dict(
                    [metric.optimize_threshold()])

                # select metrics relevant for paper
                metrics_df = metrics_df[["Model name", "Image CNN",
                                        "# Parameters (without Image CNN)",
                                         "Total parameters", "lr", "Batch size",
                                         "# Epochs", "AUC-ROC", "AUC-PRC",
                                         "Precision", "Recall", "F-Score",
                                         "Threshold F-Score", "Accuracy F-Score"]]
                metrics_df["lr"] = metrics_df["lr"].astype(float)

                print(metrics_df.to_markdown(index=False))
                final_accs.append(metrics_df["Accuracy F-Score"])
                k += 1
        print(final_accs)
