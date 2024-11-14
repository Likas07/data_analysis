import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidfuzz import fuzz
from tqdm import tqdm
import re
from unidecode import unidecode  # For accent removal

# Define file paths and constants
IMPORTS_DATABASE_FILE = 'imports_database.xlsx'
INGREDIENTS_FILE = 'tb_ingredients.xlsx'
FORMULATED_NAMES_FILE = 'list_formulatednames.xlsx'
MATCHED_DATA_FILE = 'matched_imports_data_combined.xlsx'

# Load data
imports_db = pd.read_excel(IMPORTS_DATABASE_FILE, sheet_name='TRADEMARK')
ingredients = pd.read_excel(INGREDIENTS_FILE, header=None)[0].tolist()
formulated_names = pd.read_excel(FORMULATED_NAMES_FILE, header=None)[0].tolist()

# Step 1: Preprocess Data
# Filter rows based on DataSourceName
imports_db = imports_db[imports_db['DataSourceName'] == 'Base Brasil - Logcomex']

# Remove duplicates, handle missing values, and normalize text
imports_db.drop_duplicates(inplace=True)
imports_db.dropna(subset=["Dimension Value", "Owner", "NCM"], inplace=True)


# Normalize `Dimension Value`: lowercase, remove accents, remove punctuation, and split by whitespace
def clean_dimension_value(text):
    text = unidecode(text.lower())  # Convert to lowercase and remove accents
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split by whitespace
    return " ".join(words)  # Rejoin words with a single space


imports_db['Dimension Value'] = imports_db['Dimension Value'].apply(clean_dimension_value)

# Combine ingredients and formulated names for reference, applying the same text normalization
reference_text = [unidecode(str(i).lower()) for i in ingredients + formulated_names if isinstance(i, str)]
reference_text = [" ".join(re.sub(r'[^\w\s]', '', item).split()) for item in
                  reference_text]  # Ensure consistent whitespace

# Custom stop words, including frequent words in "Dimension Value"
custom_stopwords = ['de', 'a', 'o', 'e', 'do', 'da', 'dos', 'das', 'em', 'para', 'com', 'por', 'no', 'na', 'nos', 'nas',
                    'aditivo', 'acido', 'solucao', 'oleo', 'base', 'produto']  # Add common, non-specific terms

# Step 2: TF-IDF Vectorization with single-word matching
vectorizer = TfidfVectorizer(stop_words=custom_stopwords, ngram_range=(1, 1))  # Match single words only
dimension_value_vectors = vectorizer.fit_transform(imports_db['Dimension Value'])
reference_vectors = vectorizer.transform(reference_text)

# Cosine similarity threshold increased for strict matching
cosine_threshold = 0.5  # Increase threshold for stricter matches


# Step 3: Batch and Parallelize Cosine Similarity Calculations
def calculate_similarity_batch(batch_start, batch_end):
    """Calculate cosine similarity for a batch of dimension values against all references."""
    batch_vectors = dimension_value_vectors[batch_start:batch_end]
    similarity_batch = cosine_similarity(batch_vectors, reference_vectors)
    similarity_batch[similarity_batch < cosine_threshold] = 0
    return csr_matrix(similarity_batch)


batch_size = 500
batches = range(0, dimension_value_vectors.shape[0], batch_size)
similarity_matrices = []

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(calculate_similarity_batch, i, min(i + batch_size, dimension_value_vectors.shape[0])): i
               for i in batches}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating cosine similarity in batches"):
        similarity_matrices.append(future.result())

similarity_matrix = vstack(similarity_matrices)


# Step 4: Strict Fuzzy Matching with token_set_ratio
def refine_matches(batch_indices, similarity_matrix, dimension_values):
    """Refine cosine similarity matches with RapidFuzz for a batch of dimension values."""
    refined_matches = []
    for row_idx in batch_indices:
        cosine_matches = similarity_matrix[row_idx].nonzero()[1]
        if cosine_matches.size == 0:
            refined_matches.append("")
            continue
        matches = []
        for idx in cosine_matches:
            ref_text = reference_text[idx]
            rf_score = fuzz.token_set_ratio(dimension_values[row_idx], ref_text)
            if rf_score >= 90:  # Only consider high fuzzy matches
                combined_score = 0.9 * similarity_matrix[row_idx, idx] + 0.1 * (rf_score / 100)
                matches.append((ref_text, combined_score))

        # Sort matches, remove duplicates, and keep the top 3 unique matches
        top_matches = sorted(set(matches), key=lambda x: x[1], reverse=True)[:3]
        refined_matches.append(", ".join([match[0] for match in top_matches]))
    return refined_matches


# Parallelized Refinement
refined_results = []
dimension_values = imports_db['Dimension Value'].tolist()
batch_indices = range(0, similarity_matrix.shape[0], batch_size)

with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(refine_matches, list(range(i, min(i + batch_size, len(dimension_values)))), similarity_matrix,
                        dimension_values): i for i in batch_indices}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Refining matches with RapidFuzz"):
        refined_results.extend(future.result())


# Step 5: Product Classification and Final Output Preparation
def classify_product(ncm):
    """Classify product based on the first digit of the NCM code."""
    return "TECHNICAL" if str(ncm).startswith("2") else "FORMULATED" if str(ncm).startswith("3") else "UNKNOWN"


imports_db['Matching Words'] = refined_results
imports_db['Product Type'] = imports_db['NCM'].apply(classify_product)

# Step 6: Sorting the Output
# First, rows with matches in 'Matching Words' will come first, followed by those without matches
imports_db['Has Match'] = imports_db['Matching Words'] != ""  # Create a flag for sorting
sorted_data = imports_db.sort_values(by='Has Match', ascending=False).drop(columns=['Has Match'])

# Step 7: Select Relevant Columns for Final Output
matched_data = sorted_data[['Matching Words', 'Owner', 'Product Type']]

# Step 8: Save Results to Excel
matched_data.to_excel(MATCHED_DATA_FILE, index=False)
print(f'Matching complete. Results saved to {MATCHED_DATA_FILE}')
