import argparse

import numpy as np
from tensorflow.keras.applications.convnext import ConvNeXtBase	
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import pathlib

# You need to run kaggle competitions download -c imagenet-object-localization-challenge followed by unzip 
# to get the dataset for processing (or get our preprocessed version)
parser = argparse.ArgumentParser(description='Download and process ImageNet.')
parser.add_argument('--dataset_path', type=str, help='Path to the downloaded ImageNet dataset (from kaggle).')
args = parser.parse_args()

# Load ConvNextBase model
embedding_model = ConvNeXtBase(weights='imagenet', include_top=False, pooling="avg")
embedding_model.summary()

def get_image_embeddings(image_paths, model):
    images = []

    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)

    images = np.vstack(images)
    all_embeddings = []
    for i in range(0, len(images), 64):
        all_embeddings.append(model.predict_on_batch(images[i: i + 64]))
    return all_embeddings

pathlib.Path('data/imagenet').mkdir(parents=True, exist_ok=True) 
output_embeddings_path = 'data/imagenet/imagenet.npy'
output_labels_path = 'data/imagenet/imagenet_gt.npy'

embeddings = []
labels = []

for root, dirs, files in os.walk(args.dataset_path):
    if "train" not in root:
        continue
    filenames = []
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, filename)
            filenames.append(image_path)
    if len(filenames) > 1:
        embeddings += get_image_embeddings(filenames, embedding_model)
        labels += [os.path.basename(root) for _ in range(len(filenames))]
        print(len(labels), sum([len(e) for e in embeddings]), embeddings[-1].shape)

embeddings = np.vstack(embeddings)
_, labels = np.unique(labels, return_inverse=True)

np.save(output_embeddings_path, embeddings)
np.save(output_labels_path, labels)
