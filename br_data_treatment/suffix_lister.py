import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack
from concurrent.futures import ProcessPoolExecutor, as_completed
from rapidfuzz import fuzz
from tqdm import tqdm
import re
from unidecode import unidecode
import spacy

# Load spaCy's pre-trained Portuguese model for NER
nlp = spacy.load("pt_core_news_sm")

# Define file paths
file_paths = {
    'imports_database': 'imports_database.xlsx',
    'ingredients': 'tb_ingredients.xlsx',
    'formulated_names': 'list_formulatednames.xlsx',
    'disregard_companies': 'disregard_company.xlsx',
    'output_file': 'matched_imports_data_combined.xlsx'
}

# Initialize progress bar for the entire script
total_steps = 10  # Total steps in the process
progress_bar = tqdm(total=total_steps, desc="Starting script...")

# Step 1: Load data
progress_bar.set_description("Loading data")
imports_db = pd.read_excel(file_paths['imports_database'], sheet_name='TRADEMARK')
ingredients = pd.read_excel(file_paths['ingredients'], header=None)[0].dropna().tolist()
formulated_names = pd.read_excel(file_paths['formulated_names'], header=None)[0].dropna().tolist()
disregard_companies = pd.read_excel(file_paths['disregard_companies'], sheet_name='DISREGARD')['Company'].dropna().tolist()
progress_bar.update(1)

# Step 2: Filter imports data by conditions
progress_bar.set_description("Filtering data")
imports_db = imports_db[imports_db['DataSourceName'] == 'Base Brasil - Logcomex']
imports_db = imports_db[~imports_db['Owner'].isin(disregard_companies)]
imports_db.drop_duplicates(inplace=True)
imports_db.dropna(subset=["Dimension Value", "Owner", "NCM"], inplace=True)
progress_bar.update(1)

# Step 3: Text normalization function
progress_bar.set_description("Normalizing text")
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unidecode(text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

# Apply text cleaning
imports_db['Normalized Dimension Value'] = imports_db['Dimension Value'].apply(clean_text)
reference_text = [clean_text(item) for item in ingredients + formulated_names]
progress_bar.update(1)

# Step 4: Keyword Pre-Filter
progress_bar.set_description("Applying keyword pre-filter")
# Generate a set of unique keywords from reference_text
reference_keywords = set(word for text in reference_text for word in text.split())

# Define the pre-filter function
def keyword_pre_filter(dimension_value, keywords):
    for keyword in keywords:
        if keyword in dimension_value:
            return True
    return False

# Apply the keyword pre-filter to flag potential matches
imports_db['Has Keyword Match'] = imports_db['Normalized Dimension Value'].apply(
    lambda x: keyword_pre_filter(x, reference_keywords)
)
progress_bar.update(1)

# Filter rows for further processing based on keyword matches
matched_rows = imports_db[imports_db['Has Keyword Match']].copy()
unmatched_rows = imports_db[~imports_db['Has Keyword Match']].copy()
print(f"Rows flagged by keyword pre-filter: {len(matched_rows)}")
print(f"Rows not flagged by keyword pre-filter: {len(unmatched_rows)}")

# Step 5: Batch entity extraction with spaCy
progress_bar.set_description("Extracting entities")
def batch_extract_entities(text_list, batch_size=1000):
    entities_list = []
    for doc in nlp.pipe(text_list, batch_size=batch_size):
        entities = " ".join([ent.text for ent in doc.ents]) if doc.ents else ""
        entities_list.append(entities)
    return entities_list

# Extract entities in batches
matched_rows['Entities'] = batch_extract_entities(matched_rows['Normalized Dimension Value'].tolist())
reference_entities = batch_extract_entities(reference_text)
progress_bar.update(1)

# Step 6: Vectorization with TF-IDF
progress_bar.set_description("Vectorizing text")
custom_stopwords = [
    'de', 'a', 'o', 'e', 'do', 'da', 'dos', 'das', 'em', 'para', 'com', 'por', 'no', 'na', 'nos', 'nas',
    'aditivo', 'acido', 'solucao', 'oleo', 'base', 'produto'
]

vectorizer = TfidfVectorizer(stop_words=custom_stopwords, ngram_range=(1, 3))
dimension_value_vectors = vectorizer.fit_transform(matched_rows['Entities'])
reference_vectors = vectorizer.transform(reference_entities)
progress_bar.update(1)

# Step 7: Cosine similarity calculation
progress_bar.set_description("Calculating cosine similarity")
cosine_threshold = 0.4
batch_size = 500

def calculate_similarity_batch(batch_start, batch_end):
    batch_vectors = dimension_value_vectors[batch_start:batch_end]
    similarity_batch = cosine_similarity(batch_vectors, reference_vectors)
    similarity_batch[similarity_batch < cosine_threshold] = 0
    return csr_matrix(similarity_batch)

batches = range(0, dimension_value_vectors.shape[0], batch_size)
similarity_matrices = []

with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(calculate_similarity_batch, i, min(i + batch_size, dimension_value_vectors.shape[0])): i
        for i in batches
    }
    for future in as_completed(futures):
        similarity_matrices.append(future.result())
progress_bar.update(1)

similarity_matrix = vstack(similarity_matrices)

# Step 8: Refine Matches with Fuzzy Matching, capturing the match from Dimension Value
progress_bar.set_description("Refining matches with fuzzy matching")
def refine_matches(indices, similarity_matrix, dimension_values):
    refined_matches = []
    for row_idx in indices:
        cosine_matches = similarity_matrix[row_idx].nonzero()[1]
        if cosine_matches.size == 0:
            refined_matches.append("")
            continue

        matches = []
        for idx in cosine_matches:
            ref_text = reference_entities[idx]
            rf_score = fuzz.token_set_ratio(dimension_values[row_idx], ref_text)
            combined_score = 0.6 * similarity_matrix[row_idx, idx] + 0.4 * (rf_score / 100)

            # Capture the exact match from Dimension Value
            if rf_score >= 90:  # Only consider high fuzzy matches
                words = ref_text.split()
                for word in words:
                    if word in dimension_values[row_idx]:
                        matches.append((word, combined_score))

        # Sort matches by combined score and keep the top match
        if matches:
            best_match = sorted(matches, key=lambda x: x[1], reverse=True)[0][0]
        else:
            best_match = ""
        refined_matches.append(best_match)

    return refined_matches

# Apply refined matching in parallel
refined_results = []
dimension_values = matched_rows['Entities'].tolist()
batch_indices = range(0, similarity_matrix.shape[0], batch_size)

with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(refine_matches, list(range(i, min(i + batch_size, len(dimension_values)))), similarity_matrix,
                        dimension_values): i
        for i in batch_indices
    }
    for future in as_completed(futures):
        refined_results.extend(future.result())
progress_bar.update(1)

# Step 9: Product Classification
progress_bar.set_description("Classifying products")
def classify_product(ncm):
    return "TECHNICAL" if str(ncm).startswith("2") else "FORMULATED" if str(ncm).startswith("3") else "UNKNOWN"

matched_rows['Matching Words'] = refined_results
matched_rows['Product Type'] = matched_rows['NCM'].apply(classify_product)
progress_bar.update(1)

# Step 10: Save output
progress_bar.set_description("Saving results")
output_data = pd.concat([matched_rows, unmatched_rows], ignore_index=True)
matched_data = output_data[['Dimension Value', 'Owner', 'Matching Words', 'Product Type']]
matched_data.to_excel(file_paths['output_file'], index=False)
progress_bar.update(1)

# Calculate and print the percentage of rows with a match
total_rows = matched_data.shape[0]
matched_rows_count = matched_data['Matching Words'].apply(bool).sum()
match_percentage = (matched_rows_count / total_rows) * 100
print(f"Matching complete. Results saved to {file_paths['output_file']}")
print(f"Percentage of rows with a match: {match_percentage:.2f}%")

progress_bar.close()
