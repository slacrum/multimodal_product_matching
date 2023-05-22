import os
import tarfile
import shutil
import requests
import gzip
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ast import literal_eval

def download_abo(path):
    os.makedirs(path, exist_ok=True)

    print("Downloading listings...")
    _download("https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar", path)
    
    print("Downloading images...")
    _download("https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar", path)

    print("Extracting listings...")
    with tarfile.open(os.path.join(path, "abo-listings.tar")) as f:
        f.extractall(path)

    print("Extracting images...")
    with tarfile.open(os.path.join(path, "abo-images-small.tar")) as f:
        f.extractall(path)

    print("Done")

def _download(url, dest):
    with requests.get(url, allow_redirects=True, stream=True) as r:
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))

        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length)as raw:
            with open(os.path.join(dest, url.split("/")[-1]), 'wb') as f:
                shutil.copyfileobj(raw, f)

def preprocess_data(path, alt_augment=True, random_deletion=True):
    print("Loading images...")
    with gzip.open(os.path.join(path,'images/metadata/images.csv.gz')) as f:
        images_meta = pd.read_csv(f)

    print("Loading texts...")
    if not os.path.exists(os.path.join(path, "listings/listings.csv")):
        print("Merging listings... (this may take a while)")
        json_pattern = os.path.join(path,'listings/metadata/listings_*.json.gz')
        file_list = glob.glob(json_pattern)
        dfs = []

        for f in file_list:
            with gzip.open(f) as f2:
                data = pd.read_json(f2, lines=True)
                print(f, data.shape)
                for i, row in tqdm(data.iterrows(), total=data.shape[0]):
                    dfs2 = []
                    for k in row.keys():
                        if (type(row[k]) is list):
                            if (type(row[k][0]) is dict):
                                dfs2.append(pd.json_normalize(row[k][0]).add_prefix(k + "."))
                            else:
                                dfs2.append(pd.DataFrame({k: [row[k]]}))
                        else:
                            dfs2.append(pd.DataFrame({k: [row[k]]}))
                    dfs.append(dfs2)
        dfs_1 = []

        for df in tqdm(dfs):
            dfs_1.append(pd.concat(df, axis=1))

        dfs_2 = pd.concat(dfs_1)

        print("Exporting concatenated listings...")
        dfs_2.reset_index(drop=True, inplace=True)
        dfs_2.to_csv(os.path.join(path, "listings/listings.csv"))

    print("Found listings.csv, using that instead.")
    dfs = pd.read_csv(os.path.join(path, "listings/listings.csv"))
    dfs = dfs.drop(['Unnamed: 0'], axis=1)

    if alt_augment:
        print("Performing augmentation with alternative product images...")
        dfs["other_image_id"] = dfs["other_image_id"].fillna("[]")
        dfs["other_image_id"] = dfs["other_image_id"].apply(literal_eval)
        dfs_3 = dfs.explode(["other_image_id"])
        dfs_3["main_image_id"] = dfs_3["other_image_id"]
        dfs = pd.concat([dfs, dfs_3])
        dfs.reset_index(drop=True, inplace=True)

    print("Creating ground truth...")
    listings_new = dfs[["item_keywords.value", "brand.value", "item_id", "item_name.language_tag", "item_name.value", "product_type.value", "product_description.value", "main_image_id"]]

    print("Merging images and texts...")
    ground_truth = listings_new.merge(images_meta, left_on='main_image_id', right_on='image_id')
    ground_truth["item_keywords.value2"] = ground_truth["item_keywords.value"]
    ground_truth["item_id2"] = ground_truth["item_id"]
    ground_truth["item_name.value2"] = ground_truth["item_name.value"]
    ground_truth["label"] = 1

    print("Creating false samples/complement...")
    false_samples = ground_truth.apply(np.random.permutation, axis=0)
    false_samples["label"] = 0

    print("Merging ground truth and complement...")
    dataset = pd.concat([ground_truth, false_samples])
    dataset_final = dataset.sample(frac=1, axis=0).reset_index(drop=True)
    dataset_final = dataset_final.drop(["main_image_id", "image_id", "height", "width", "product_description.value"], axis=1)
    dataset_final = dataset_final.loc[(dataset_final['item_name.language_tag'] == "en_US") | (dataset_final['item_name.language_tag'] == "en_GB") | (dataset_final['item_name.language_tag'] == "en_IN")]
    dataset_final = dataset_final.drop(["brand.value", "item_name.language_tag"], axis=1)
    dataset_final = dataset_final.reset_index(drop=True)

    if random_deletion:
        print("Performing random deletion...")
        dataset_final["item_keywords.value"] = dataset_final["item_keywords.value"].sample(frac=.5)
        dataset_final["item_id"] = dataset_final["item_id"].sample(frac=.5)
        dataset_final["item_name.value"] = dataset_final["item_name.value"].sample(frac=.5)

        dataset_final["item_keywords.value2"] = dataset_final["item_keywords.value2"].sample(frac=.5)
        dataset_final["item_id2"] = dataset_final["item_id2"].sample(frac=.5)
        dataset_final["item_name.value2"] = dataset_final["item_name.value2"].sample(frac=.5)

        dataset_final = dataset_final.fillna("")
    else:
        dataset_final = dataset_final.dropna()

    print("Concatenating attributes into description columns...")
    dataset_final["description"] = dataset_final["item_keywords.value"] + dataset_final["item_id"] + dataset_final["item_name.value"]
    dataset_final["description2"] = dataset_final["item_keywords.value2"] + dataset_final["item_id2"] + dataset_final["item_name.value2"]
    dataset_final = dataset_final[["description", "description2", "path", "label", "product_type.value"]]

    print("Finishing up...")
    dataset_final = dataset_final.drop(np.where((dataset_final['description'] == '') | (dataset_final['description2'] == ''))[0])
    dataset_final["description"] = dataset_final["description"].str.lower()
    dataset_final["description2"] = dataset_final["description2"].str.lower()
    dataset_final["description"] = dataset_final['description'].str.replace(r'[^\x00-\x7F]+', '')
    dataset_final["description2"] = dataset_final['description2'].str.replace(r'[^\x00-\x7F]+', '')

    print("Exporting to CSV...")
    dataset_final.to_csv(os.path.join(path, "data.csv"))

    print("Done")