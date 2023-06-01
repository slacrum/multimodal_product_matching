import os
import gzip
import glob
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from data_loader.dataset import Dataset


class ABO(Dataset):
    """
    Class to implement the Amazon Berkeley Objects dataset,
    as described in Collins et al., 2022 (https://arxiv.org/abs/2110.06199)
    """

    def __init__(
            self, path,
            urls=[
                "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar",
                "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"],
            download=True, extract=True, preprocess=True, alt_augment=False,
            random_deletion=True, export_csv=True):
        self.alt_augment = alt_augment
        super().__init__(path, urls, download, extract,
                         preprocess, random_deletion, export_csv)

    def _load_imgs(self):
        with gzip.open(os.path.join(self.path, 'images/metadata/images.csv.gz')) as f:
            return pd.read_csv(f)

    def _load_txts(self):
        if not (
            os.path.exists(
                os.path.join(self.path, "listings/listings.csv.gz"))
            or os.path.exists(
                os.path.join(self.path, "listings/listings.csv"))):
            print("Merging listings... (this may take a while)")
            json_pattern = os.path.join(
                self.path, 'listings/metadata/listings_*.json.gz')
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
                                    dfs2.append(pd.json_normalize(
                                        row[k][0]).add_prefix(k + "."))
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
            dfs_2.to_csv(os.path.join(self.path, "listings/listings.csv"))

        print("Importing listings CSV...")
        if os.path.exists(os.path.join(self.path, "listings/listings.csv.gz")):
            with gzip.open(os.path.join(self.path, "listings/listings.csv.gz")) as f:
                dfs = pd.read_csv(f, dtype=object)
        elif os.path.exists(os.path.join(self.path, "listings/listings.csv")):
            dfs = pd.read_csv(os.path.join(
                self.path, "listings/listings.csv"), dtype=object)

        dfs = dfs.drop(['Unnamed: 0'], axis=1)

        if self.alt_augment:
            print("Performing augmentation with alternative product images...")
            dfs["other_image_id"] = dfs["other_image_id"].fillna("[]")
            dfs["other_image_id"] = dfs["other_image_id"].apply(literal_eval)
            dfs_3 = dfs.explode(["other_image_id"])
            dfs_3["main_image_id"] = dfs_3["other_image_id"]
            dfs = pd.concat([dfs, dfs_3])
            dfs.reset_index(drop=True, inplace=True)

        dfs = dfs.loc[(dfs['item_name.language_tag'] == "en_US") |
                      (dfs['item_name.language_tag'] == "en_GB") |
                      (dfs['item_name.language_tag'] == "en_IN")]
        dfs.reset_index(drop=True, inplace=True)

        dfs["product_type"] = dfs["product_type.value"]

        dfs = dfs[["item_keywords.value", "item_id",
                   "item_name.value", "product_type", "main_image_id"]]

        return dfs
