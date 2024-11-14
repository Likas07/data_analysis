import pandas as pd
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
    'output_file': 'matched_imports_data_combined_no_similarity.xlsx'
}

# Initialize progress bar for the entire script
total_steps = 8  # Adjusted for the reduced features
progress_bar = tqdm(total=total_steps, desc="Starting script...")

# Step 1: Load data
progress_bar.set_description("Loading data")
imports_db = pd.read_excel(file_paths['imports_database'], sheet_name='TRADEMARK')
ingredients = pd.read_excel(file_paths['ingredients'], header=None)[0].dropna().tolist()
formulated_names = pd.read_excel(file_paths['formulated_names'], header=None)[0].dropna().tolist()
disregard_companies = pd.read_excel(file_paths['disregard_companies'], sheet_name='DISREGARD')[
    'Company'].dropna().tolist()
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

# Step 4: Keyword Pre-Filter Preparation
progress_bar.set_description("Preparing keyword pre-filter")
# Generate a set of unique keywords from reference_text, filtering out common or short words
reference_keywords = set(
    word for text in reference_text for word in text.split()
    if len(word) > 2 and word not in {'produto', 'com', 'base', 'outros'}  # Customize with additional common words
)

# Compile a regex pattern that matches any of the keywords as whole words (non-capturing group to avoid warnings)
keyword_pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in reference_keywords) + r')\b'
progress_bar.update(1)

# Step 5: Batch-wise Vectorized Keyword Matching in Dimension Value
progress_bar.set_description("Applying keyword pre-filter in batches")

# Define batch size
batch_size = 1000
num_batches = (len(imports_db) // batch_size) + 1  # Calculate the number of batches needed

# Initialize an empty list to store the results
keyword_matches = []

for i in tqdm(range(num_batches), desc="Processing batches for keyword matching"):
    # Select the batch
    batch_df = imports_db.iloc[i * batch_size:(i + 1) * batch_size].copy()

    # Vectorized regex matching within each batch using .loc[] to avoid SettingWithCopyWarning
    batch_df.loc[:, 'Has Keyword Match'] = batch_df['Normalized Dimension Value'].str.contains(keyword_pattern,
                                                                                               regex=True)

    # Collect results from each batch
    keyword_matches.extend(batch_df['Has Keyword Match'].tolist())

# Assign the results back to the main DataFrame
imports_db['Has Keyword Match'] = keyword_matches
progress_bar.update(1)

# Filter rows for further processing based on keyword matches
matched_rows = imports_db[imports_db['Has Keyword Match']].copy()
unmatched_rows = imports_db[~imports_db['Has Keyword Match']].copy()
print(f"Rows flagged by keyword pre-filter: {len(matched_rows)}")
print(f"Rows not flagged by keyword pre-filter: {len(unmatched_rows)}")

# Step 6: Batch entity extraction with spaCy
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

# Step 7: Simple Keyword Matching within Dimension Value
progress_bar.set_description("Matching keywords within Dimension Value")


# Updated function to find matching whole words
def find_matching_words(dimension_value, reference_keywords):
    """Finds and returns keywords from `dimension_value` that match any of the reference keywords as whole words."""
    matches = []
    # Join all keywords into a single regex pattern with word boundaries
    pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in reference_keywords) + r')\b'
    # Find all whole word matches in the dimension_value
    found_matches = re.findall(pattern, dimension_value)
    # Deduplicate matches by converting to a set, then join them into a single string
    return ", ".join(sorted(set(found_matches)))


# Apply the matching function to extract relevant words from Dimension Value
matched_rows['Matching Words'] = matched_rows['Normalized Dimension Value'].apply(
    lambda x: find_matching_words(x, reference_keywords)
)
progress_bar.update(1)

# Step 8: Product Classification
progress_bar.set_description("Classifying products")


def classify_product(ncm):
    return "TECHNICAL" if str(ncm).startswith("2") else "FORMULATED" if str(ncm).startswith("3") else "UNKNOWN"


matched_rows['Product Type'] = matched_rows['NCM'].apply(classify_product)
progress_bar.update(1)

# Step 9: Save output
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
