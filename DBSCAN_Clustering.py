from typing import List, Dict, Set
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import glob

def load_data() -> pd.DataFrame:
    """
    Load data from CSV file and return as a pandas DataFrame.
    """
    file_paths = glob.glob('data/*.csv')
    if not file_paths:
        print('No .csv files found.')
        return
    file_path = file_paths[0]
    return pd.read_csv(file_path, on_bad_lines='skip')

def cluster_strings(strings: List[str], eps: float = 0.5, min_samples: int = 5) -> List[List[str]]:
    """
    Cluster strings using DBSCAN.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(strings).toarray()

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(vectors)

    # Create a list of lists, where each inner list represents a cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            label = 'noise'
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(strings[i])

    return list(clusters.values())

def main():
    eps = 0.2
    min_samples = 3

    data = load_data()
    strings = data['string'].tolist()
    clusters = cluster_strings(strings, eps, min_samples)

    # Print clusters
    df = pd.DataFrame({'Cluster': [i+1 for i in range(len(clusters)) for _ in range(len(clusters[i]))],
                       'String': [string for cluster in clusters for string in cluster]})
    df.to_excel('clusters.xlsx', index=False)

if __name__ == "__main__":
    main()

