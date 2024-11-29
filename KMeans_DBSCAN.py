import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    return model


# Extract the bounding box coordinates from the YOLO output
def extract_features_yolo(images, model):
    features = []
    for img in images:
        results = model(img)  # Get the results from the YOLO model
        detections = results[0].boxes

        for box in detections:
            # Check if box has the required attributes
            if hasattr(box, 'xyxy') and hasattr(box, 'conf') and hasattr(box, 'cls'):
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get coordinates
                conf = box.conf[0].item()  # Get confidence
                cls = int(box.cls[0].item())  # Get class
                w = int(x2 - x1)
                h = int(y2 - y1)
                x = int(x1)
                y = int(y1)
                
                # Append the features if all values are valid
                features.append((x, y, w, h, cls, conf))
            else:
                print("Box does not have the required attributes.")
                print(f"Box: {box}")

    return features

# Uses PCA to reduce the number of variables but keeps information almost perfectly intact
def reduce_dimensionality(features, n_components=50):
    if len(features) == 0:
        raise ValueError("No features to reduce.")
    
    n_features = features.shape[1]
    n_components = min(n_components, n_features)
    
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

# Uses kmeans and dbscan from scikit learn to produce the clusters
def cluster_features(features, n_clusters=5):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(normalized_features)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(normalized_features)

    return kmeans_labels, dbscan_labels

# Displays the images in the clusters
def show_all_clusters(image_paths, cluster_labels, n_clusters):
    plt.figure(figsize=(20, 20))
    total_images_to_show = 0

    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if not cluster_indices:
            continue
        
        num_samples = min(len(cluster_indices), 10)
        for i in range(num_samples):
            idx = cluster_indices[i]
            if idx < len(image_paths):
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(n_clusters, 10, total_images_to_show + i + 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Cluster {cluster_id}, Image {idx}")
        
        total_images_to_show += num_samples

    plt.tight_layout()
    plt.show()

def main():
    weights_path = r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\runs\detect\train40\weights\best.pt'
    model = load_yolo_model(weights_path)

    image_dir = r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\stanford-car-yolov5s-1\test\images'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                   
    images = [cv2.imread(img_path) for img_path in image_paths]

    features = extract_features_yolo(images, model)

    bounding_boxes = np.array([f[:4] for f in features])  # Extract only the bounding box coordinates

    if bounding_boxes.size == 0:
        print("No objects detected. Exiting.")
        return

    if len(bounding_boxes) < 2:
        print("Not enough bounding boxes for PCA. Exiting.")
        return

    reduced_features = reduce_dimensionality(bounding_boxes)

    n_clusters = 5
    kmeans_labels, dbscan_labels = cluster_features(reduced_features, n_clusters)

    show_all_clusters(image_paths, kmeans_labels, n_clusters)

    unique_dbscan_labels = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    show_all_clusters(image_paths, dbscan_labels, unique_dbscan_labels)

if __name__ == "__main__":
    main()