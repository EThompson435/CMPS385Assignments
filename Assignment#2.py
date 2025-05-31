import numpy as np
import cv2
import os
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # NEW: tqdm for progress bar!

# 1. Feature extraction
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 32))  # Resize to standard size
    img = img / 255.0  # Normalize pixel values
    return img.flatten()

# 2. Load dataset
def load_dataset(images_folder):
    X, y = [], []

    folder_names = [f for f in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, f))]
    folder_names = sorted(folder_names, key=lambda x: int(x))  # Sort folder names numerically

    for folder_name in tqdm(folder_names, desc="Loading folders"):
        folder_path = os.path.join(images_folder, folder_name)

        folder_num = int(folder_name)

        # Numbers (0-9)
        if folder_num <= 9:
            label = str(folder_num)
        else:
            # Letters (folder 10 and above)
            char_index = (folder_num - 10) // 2
            is_upper = (folder_num - 10) % 2
            letter = chr(ord('a') + char_index)
            label = letter.upper() if is_upper else letter.lower()

        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        for filename in tqdm(image_files, desc=f"Loading images from folder {folder_name}", leave=False):
            img_path = os.path.join(folder_path, filename)

            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

# 3. Train HMMs
def train_hmms(X_train, y_train):
    models = {}
    labels = np.unique(y_train)
    for label in tqdm(labels, desc="Training HMMs"):
        X_label = X_train[y_train == label]
        lengths = [len(x) for x in X_label]
        X_concat = np.vstack(X_label)

        model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
        model.fit(X_concat.reshape(-1, 1), lengths)
        models[label] = model
    return models

# 4. Predict
def predict(models, X_test):
    y_pred = []
    for sample in tqdm(X_test, desc="Predicting"):
        scores = {}
        sample = sample.reshape(-1, 1)
        for label, model in models.items():
            try:
                score = model.score(sample)
                scores[label] = score
            except:
                scores[label] = float('-inf')
        best_label = max(scores, key=scores.get)
        y_pred.append(best_label)
    return y_pred

# 5. Main script
def main():
    # Update path to Downloads
    images_folder = 'C:/Users/Ethan Thompson/Downloads/V0.3/data'

    X, y = load_dataset(images_folder)

    if len(X) == 0:
        print("No images loaded! Check paths.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_hmms(X_train, y_train)

    y_pred = predict(models, X_test)

    accuracy = np.mean(np.array(y_pred) == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
