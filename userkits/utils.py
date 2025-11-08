import os
import cv2
import tqdm
import numpy as np
import pandas as pd

def load_train_data(data_dir='../train_data', half=False):
    X = []
    y = []
    for biome_name in tqdm.tqdm(os.listdir(data_dir), desc="Loading train data"):
        biome_path = os.path.join(data_dir, biome_name)
        if not os.path.isdir(biome_path):
            continue
        for img_name in tqdm.tqdm(os.listdir(biome_path), desc=f"Processing {biome_name}", leave=False):
            img_path = os.path.join(biome_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            if half:
                img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            X.append(img)
            y.append(biome_name)
    return X, y

def load_eval_data(data_dir='../eval_data', half=False):
    X = []
    file_ids = []
    for img_name in tqdm.tqdm(os.listdir(data_dir), desc="Loading eval data"):
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if half:
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        X.append(img)
        file_ids.append(img_name)
    return X, file_ids

def save_predictions(predictions, file_ids, output_file='predictions.csv'):
    df = pd.DataFrame({"id": file_ids, "prediction": predictions})
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    return