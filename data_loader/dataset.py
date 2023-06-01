import os
import tarfile
import shutil
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(self, path, urls, download, extract, preprocess,
                 random_deletion, export_csv):
        self.path = path
        if download:
            self.urls = urls
            self._download_dataset(extract=extract)
        if preprocess:
            self._preprocess_data(
                random_deletion=random_deletion, export_csv=export_csv)

    def _download_dataset(self, extract=True):
        os.makedirs(self.path, exist_ok=True)

        for url in self.urls:
            file = url.split("/")[-1]
            print(f"Downloading {file}...")
            if not os.path.exists(os.path.join(self.path, file)):
                self._download(url, self.path)
            else:
                print(f"{file} already exists.")

            if extract:
                print(f"Extracting {file}...")
                if file.endswith(".tar"):
                    with tarfile.open(os.path.join(self.path, file)) as f:
                        f.extractall(self.path)

    def _download(self, url, dest):
        with requests.get(url, allow_redirects=True, stream=True) as r:
            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))

            # implement progress bar via tqdm
            with tqdm.wrapattr(r.raw, "read", total=total_length)as raw:
                with open(os.path.join(dest, url.split("/")[-1]), 'wb') as f:
                    shutil.copyfileobj(raw, f)

    def _preprocess_data(self, random_deletion=True, export_csv=True):
        print("Loading images...")
        imgs = self._load_imgs()

        print("Loading texts...")
        txts = self._load_txts()

        ground_truth = txts.merge(
            imgs, left_on=txts.columns[-1], right_on=imgs.columns[0])
        for col in txts.columns[:-2]:
            ground_truth[col + "2"] = ground_truth[col]
        ground_truth["label"] = 1

        print("Creating false samples/complement...")
        false_samples = ground_truth.apply(np.random.permutation, axis=0)
        false_samples["label"] = 0

        print("Merging ground truth and complement...")
        dataset = pd.concat([ground_truth, false_samples])
        dataset_final = dataset.sample(frac=1, axis=0).reset_index(drop=True)
        dataset_final.reset_index(drop=True, inplace=True)

        if random_deletion:
            print("Performing random deletion...")
            for col in txts.columns[:-2]:
                dataset_final[col] = dataset_final[col].sample(frac=.5)
                dataset_final[col + "2"] = dataset_final[col +
                                                         "2"].sample(frac=.5)
            dataset_final = dataset_final.fillna("")
        else:
            dataset_final = dataset_final.dropna()

        print("Concatenating attributes into description columns...")
        dataset_final["description"] = dataset_final[txts.columns[:-2]
                                                     ].apply("".join, axis=1)
        dataset_final["description2"] = dataset_final[txts.columns[:-
                                                                   2] + "2"].apply("".join, axis=1)
        dataset_final = dataset_final[[
            "description", "description2", "path", "label", "product_type"]]

        print("Finishing up...")
        dataset_final = dataset_final.drop(
            np.where(
                (dataset_final['description'] == '') |
                (dataset_final['description2'] == ''))[0])
        dataset_final["description"] = dataset_final["description"].str.lower()
        dataset_final["description2"] = dataset_final["description2"].str.lower()
        dataset_final["description"] = dataset_final['description'].str.replace(
            r'[^\x00-\x7F]+', '', regex=True)
        dataset_final["description2"] = dataset_final['description2'].str.replace(
            r'[^\x00-\x7F]+', '', regex=True)
        dataset_final = dataset_final.dropna()
        dataset_final = dataset_final.reset_index(drop=True)

        if export_csv:
            print("Exporting to CSV...")
            dataset_final.to_csv(os.path.join(self.path, "data.csv"))

        print("Data processing complete")
        self.data = dataset_final

    def _load_imgs(self):
        # return dataframe with cols in the format [img_id, col1, col2,...]
        pass

    def _load_txts(self):
        # return dataframe with cols in the format [col1, col2,..., product_type, img_id]
        pass
