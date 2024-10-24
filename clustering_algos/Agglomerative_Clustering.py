from typing import List, Dict, Set
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
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

def cluster_strings(strings: List[str], n_clusters: int) -> List[List[str]]:
    """
    Cluster strings using hierarchical clustering.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(strings).toarray()

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(vectors).tolist()
    clusters = [[strings[i] for i in range(len(strings)) if clusters[i] == j] for j in range(n_clusters)]
    return clusters

def main():
    threshold = 0.9
    n_clusters = 8

    data = load_data()
    strings = data['string'].tolist()
    clusters = cluster_strings(strings, n_clusters)

    # Print clusters
    df = pd.DataFrame({'Cluster': [i+1 for i in range(len(clusters)) for _ in range(len(clusters[i]))],
                       'String': [string for cluster in clusters for string in cluster]})
    df.to_excel('clusters.xlsx', index=False)

if __name__ == "__main__":
    main()

