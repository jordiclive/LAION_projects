import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import os
def merge(dataset_items):
    for i, dataset_item in enumerate(dataset_items):
        if i == 0:
            df = pd.read_parquet(dataset_item)

        else:
            new_item = pd.read_parquet(dataset_item)
            df = pd.concat([df, new_item])

    df.reset_index(inplace=True, drop=True)
    return df


def merge_datasets():
    if not os.path.exists("final_dataset"):
        os.makedirs("final_dataset")
    train_sets = glob.glob("*")
    train = merge(train_sets)
    train.to_parquet("train.parquet")

    val_sets = glob.glob("val*")
    val = merge(val_sets)
    val.to_parquet("val.parquet")



def val_split():
    sets = glob.glob("*")
    for set in sets:
        train, val = train_test_split(train, test_size=100)
        train.reset_index(inplace=True, drop=True)
        val.reset_index(inplace=True, drop=True)
        set = set.split('/')[-1]
        train.to_parquet(f"train_{set}.parquet")
        val.to_parquet(f"val_{set}.parquet")

if __name__ == "__main__":
    # df = pd.read_csv("hf_datasets.csv")
    # process_datasets(df)
    merge_datasets()
