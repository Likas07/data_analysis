from typing import List, Dict, Set
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import glob

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file and return as a pandas DataFrame.
    """
    file_paths = glob.glob('data/*.csv')
    if not file_paths:
        print('No .csv files found.')
        return
    file_path = file_paths[0]
    return pd.read_csv(file_path, on_bad_lines='skip')

def create_similarity_dict(strings: List[str], threshold: int) -> Dict[str, Set[str]]:
    """
    Create dictionary of similarity relationships between strings.
    """
    similarity_dict = {}
    for i, string1 in enumerate(strings):
        similarity_dict.setdefault(string1, set())
        for string2 in strings[i+1:]:
            similarity = calculate_similarity(string1, string2)
            if similarity > threshold:
                similarity_dict.setdefault(string1, set()).add(string2)
                similarity_dict.setdefault(string2, set()).add(string1)
    return similarity_dict

def calculate_similarity(string1: str, string2: str) -> float:
    """
    Calculate the similarity between two strings using token sort ratio.
    """
    token_set1 = set(sorted(string1.split()))
    token_set2 = set(sorted(string2.split()))
    common_tokens = token_set1.intersection(token_set2)
    if len(common_tokens) == 0:
        return 0.0
    else:
        return len(common_tokens) / (len(token_set1) + len(token_set2) - len(common_tokens))

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
    n_clusters = 10

    data = load_data()
    strings = data['string'].tolist()
    similarity_dict = create_similarity_dict(strings, threshold)
    clusters = cluster_strings(strings, n_clusters)

    # Print clusters
    df = pd.DataFrame({'Cluster': [i+1 for i in range(len(clusters)) for _ in range(len(clusters[i]))],
                       'String': [string for cluster in clusters for string in cluster]})
    df.to_excel('clusters.xlsx', index=False)

if __name__ == "__main__":
    main()

