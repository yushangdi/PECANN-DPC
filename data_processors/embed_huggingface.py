import datasets
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import pandas as pd
from pathlib import Path

download_dir = ""
data_dir = ""


class Metadata:
    def __init__(self, short_name, split, text_col, label_col, multiple_per_entry):
        self.short_name = short_name
        self.split = split
        self.text_col = text_col
        self.label_col = label_col
        self.multiple_per_entry = multiple_per_entry


dataset_metadata = {
    "mteb/arxiv-clustering-p2p": Metadata(
        "arxiv-clustering-p2p", "test", "sentences", "labels", True
    ),
    "mteb/arxiv-clustering-s2s": Metadata(
        "arxiv-clustering-s2s", "test", "sentences", "labels", True
    ),
    "mteb/reddit-clustering": Metadata(
        "reddit-clustering", "test", "sentences", "labels", True
    ),
    "mteb/reddit-clustering-p2p": Metadata(
        "reddit-clustering-p2p", "test", "sentences", "labels", True
    ),
    "mteb/stackexchange-clustering": Metadata(
        "stackexchange-clustering", "test", "sentences", "labels", True
    ),
    "mteb/stackexchange-clustering-p2p": Metadata(
        "stackexchange-clustering-p2p", "test", "sentences", "labels", True
    ),
}


def embed(dataset_name):
    data = datasets.load_dataset(dataset_name, cache_dir=download_dir)

    model_name = "thenlper/gte-large"
    embedder = SentenceTransformer(model_name, cache_folder=download_dir)

    metadata = dataset_metadata[dataset_name]
    documents = []
    ground_truths = []
    for document in data[metadata.split]:
        gt = document[metadata.label_col]
        text = document[metadata.text_col]

        if not metadata.multiple_per_entry:
            gt = [gt]
            text = [text]

        ground_truths += gt
        documents += text

    ground_truths = pd.factorize(ground_truths)[0]
    print(len(documents), flush=True)
    print(len(ground_truths), flush=True)

    start = time.time()

    pool = embedder.start_multi_process_pool()
    embeddings_np = embedder.encode_multi_process(documents, pool)

    print(embeddings_np.shape)
    print(len(ground_truths))
    print(time.time() - start)

    dataset_dir = f"{data_dir}/{metadata.short_name}"
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{dataset_dir}/{metadata.short_name}.npy", embeddings_np)
    with open(f"{dataset_dir}/{metadata.short_name}.gt", "w") as f:
        for g in ground_truths:
            f.write(f"{g} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("download_dir")
    parser.add_argument("data_dir")
    args = parser.parse_args()

    download_dir = args.download_dir
    data_dir = args.data_dir

    embed(args.dataset_name)
