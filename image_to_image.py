# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import os
# from tqdm import tqdm

# weights = models.ResNet50_Weights.DEFAULT
# model = models.resnet50(weights=weights)
# model = torch.nn.Sequential(*list(model.children())[:-1])
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
# ])

# def extract_features(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         features = model(image)
#     return features.numpy().flatten()

# def get_image_paths(directory, extensions=('jpg', 'png')):
#     image_paths = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(extensions):
#                 image_paths.append(os.path.join(root, file))
#     return image_paths

# def recommend_images_based_on_image(reference_image_path, image_dir='./train_cleaned', num_images=5):
#     image_paths = get_image_paths(image_dir)
#     features = []
#     for img in tqdm(image_paths, desc="Extraction des caractéristiques"):
#         features.append(extract_features(img))
#     reference_features = extract_features(reference_image_path)

#     similarities = np.dot(features, reference_features) / (np.linalg.norm(features, axis=1) * np.linalg.norm(reference_features))
#     similar_image_indices = np.argsort(similarities)[::-1]

#     return [image_paths[i] for i in similar_image_indices[:num_images]]


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()

def get_image_paths(directory, category, extensions=('jpg', 'png')):
    image_paths = []
    category_dir = os.path.join(directory, category)
    print(f"Searching in directory: {category_dir}")  # Debug print
    for root, _, files in os.walk(category_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    print(f"Found {len(image_paths)} images in category '{category}'")  # Debug print
    return image_paths

def recommend_images_based_on_image(reference_image_path, image_dir, category, num_images=5):
    image_paths = get_image_paths(image_dir, category)
    features = []
    for img in tqdm(image_paths, desc="Extraction des caractéristiques"):
        features.append(extract_features(img))
    reference_features = extract_features(reference_image_path)

    similarities = np.dot(features, reference_features) / (np.linalg.norm(features, axis=1) * np.linalg.norm(reference_features))
    similar_image_indices = np.argsort(similarities)[::-1]

    return [image_paths[i] for i in similar_image_indices[:num_images]]

def cluster_images(image_paths, n_clusters=5):
    features = [extract_features(img) for img in image_paths]
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)
    return clusters

def visualize_clusters(image_paths, clusters):
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)
    fig, axes = plt.subplots(1, num_clusters, figsize=(15, 5))
    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        for idx in cluster_indices:
            img = Image.open(image_paths[idx])
            axes[cluster].imshow(img)
            axes[cluster].axis('off')
        axes[cluster].set_title(f'Cluster {cluster + 1}')
    plt.tight_layout()
    plt.show()
