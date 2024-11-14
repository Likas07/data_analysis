import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import joblib
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from rapidfuzz import process, fuzz
import gc
import spacy
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize progress bar for the main pipeline
total_steps = 14
progress_bar = tqdm(total=total_steps, desc="Starting model training pipeline")

# Step 1: Load Data with selective columns
logger.info("Loading data...")
trademark_data = pd.read_excel('Dimensions not registered -2024-10-24-08-00-26.xlsx', sheet_name='TRADEMARK',
                               usecols=['DataSourceName', 'Dimension Value', 'Owner', 'NCM'])
ft_trademark_data = pd.read_excel('tb_ft_trademark.xlsx', usecols=['ID', 'OR_TRADEMARK_LABEL', 'NAM_TRADEMARK_IMPORT'])
ingredient_data = pd.read_excel('tb_ingredient.xlsx')
ft_ingredients_data = pd.read_excel('tb_ft_ingredients.xlsx', usecols=['ID', 'INGREDIENT_NAME_IMPORT', 'OR_INGREDIENT_LABEL'])

# Load DISREGARD worksheet
disregard_companies = pd.read_excel('disregard_company.xlsx', sheet_name='DISREGARD', usecols=['Company'])
disregard_list = disregard_companies['Company'].dropna().tolist()
progress_bar.update(1)

# Step 2: Filter by DataSourceName and Disregard List
filtered_data = trademark_data[
    (trademark_data['DataSourceName'] == "Base Brasil - Logcomex") &
    (~trademark_data['Owner'].isin(disregard_list))
].copy()
progress_bar.update(1)

# Step 3: Parallelized Label TECHNICAL or FORMULATED based on NCM Code
def label_technical_formulated(ncm):
    if str(ncm).startswith("2"):
        return 0  # TECHNICAL
    elif str(ncm).startswith("3"):
        return 1  # FORMULATED
    else:
        return -1  # UNKNOWN or other

with ProcessPoolExecutor() as executor:
    filtered_data['Type_Label'] = list(executor.map(label_technical_formulated, filtered_data['NCM']))
filtered_data['Type_Label'] = filtered_data['Type_Label'].astype('int8')  # Memory optimization
progress_bar.update(1)

# Step 4: Parallel Fuzzy Matching for Owner with OR_TRADEMARK_LABEL
def fuzzy_match_batch(batch, target_series, scorer):
    """Function to perform fuzzy matching on a batch of records."""
    results = []
    for item in batch:
        match = process.extractOne(item, target_series, scorer=scorer)
        results.append(match[0] if match else '')
    return results

logger.info("Parallel fuzzy matching for Owner with OR_TRADEMARK_LABEL...")
batch_size = 500
owner_batches = [filtered_data['Owner'][i:i + batch_size] for i in range(0, len(filtered_data['Owner']), batch_size)]
with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(fuzzy_match_batch, batch, ft_trademark_data['OR_TRADEMARK_LABEL'], fuzz.token_sort_ratio)
        for batch in owner_batches
    ]
    owner_matches = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Owner Fuzzy Matching", leave=False):
        owner_matches.extend(future.result())
filtered_data['Trademark_Match'] = owner_matches

# Merge Trademark Data with Fuzzy Matches
filtered_data = filtered_data.merge(
    ft_trademark_data[['OR_TRADEMARK_LABEL', 'NAM_TRADEMARK_IMPORT']],
    left_on='Trademark_Match', right_on='OR_TRADEMARK_LABEL', how='left'
).drop(columns=['Trademark_Match', 'OR_TRADEMARK_LABEL'])
progress_bar.update(1)
del owner_batches, owner_matches  # Free memory
gc.collect()

# Step 5: Parallelized Fuzzy Matching for Dimension Value with INGREDIENT_NAME_IMPORT
logger.info("Parallel fuzzy matching for Dimension Value with INGREDIENT_NAME_IMPORT...")
dimension_batches = [filtered_data['Dimension Value'][i:i + batch_size] for i in range(0, len(filtered_data['Dimension Value']), batch_size)]
with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(fuzzy_match_batch, batch, ft_ingredients_data['INGREDIENT_NAME_IMPORT'], fuzz.partial_ratio)
        for batch in dimension_batches
    ]
    ingredient_matches = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Dimension Value Fuzzy Matching", leave=False):
        ingredient_matches.extend(future.result())
filtered_data['Ingredient_Match'] = ingredient_matches

# Merge Ingredient Data with Fuzzy Matches
filtered_data = filtered_data.merge(
    ft_ingredients_data[['INGREDIENT_NAME_IMPORT', 'OR_INGREDIENT_LABEL']],
    left_on='Ingredient_Match', right_on='INGREDIENT_NAME_IMPORT', how='left'
).drop(columns=['Ingredient_Match', 'INGREDIENT_NAME_IMPORT'])
progress_bar.update(1)
del dimension_batches, ingredient_matches  # Free memory
gc.collect()

# Step 6: Text Cleaning and Entity Recognition
logger.info("Cleaning and combining text fields...")
nlp = spacy.load("pt_core_news_sm")

def clean_text(text):
    text = unidecode(text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def extract_entities(text):
    doc = nlp(text)
    entities = " ".join([ent.text for ent in doc.ents])
    return entities

# Function to combine text fields (instead of lambda)
def combine_text_fields(row):
    return clean_text(
        row['Dimension Value'] + ' ' +
        row['Owner'] + ' ' +
        row.get('NAM_TRADEMARK_IMPORT', '') + ' ' +
        row.get('OR_INGREDIENT_LABEL', '')
    )

if os.path.exists("dataset_with_cleaned_text.csv"):
    logger.info("Loading preprocessed cleaned text data with entities...")
    filtered_data = pd.read_csv("dataset_with_cleaned_text.csv")
else:
    with ProcessPoolExecutor() as executor:
        # Use the regular function 'combine_text_fields' instead of lambda
        filtered_data['cleaned_description'] = list(executor.map(
            combine_text_fields, filtered_data.to_dict(orient='records')
        ))
        filtered_data['extracted_entities'] = list(executor.map(extract_entities, filtered_data['cleaned_description']))
    filtered_data.to_csv("dataset_with_cleaned_text.csv", index=False)
progress_bar.update(1)


# Step 7: Vectorization
logger.info("Vectorizing text data...")
if os.path.exists("tfidf_vectorizer.joblib"):
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    X_tfidf = vectorizer.transform(filtered_data['cleaned_description'] + ' ' + filtered_data['extracted_entities'])
else:
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), stop_words='english')
    X_tfidf = vectorizer.fit_transform(filtered_data['cleaned_description'] + ' ' + filtered_data['extracted_entities'])
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
X_combined = hstack([X_tfidf, csr_matrix(filtered_data[['Type_Label']].values)])
progress_bar.update(1)

# Step 8: Label Encoding
logger.info("Encoding target labels...")
if os.path.exists("label_encoder.joblib"):
    label_encoder = joblib.load("label_encoder.joblib")
    y = label_encoder.transform(filtered_data['NAM_TRADEMARK_IMPORT'].fillna('UNKNOWN'))
else:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(filtered_data['NAM_TRADEMARK_IMPORT'].fillna('UNKNOWN'))
    joblib.dump(label_encoder, "label_encoder.joblib")
progress_bar.update(1)

# Step 9: Train Random Forest Classifier
logger.info("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
progress_bar.update(1)

# Step 10: Evaluate the model
logger.info("Evaluating the model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
progress_bar.update(1)

# Step 11: Save the model
logger.info("Saving model and preprocessing objects...")
joblib.dump({
    'model': model,
    'vectorizer': vectorizer,
    'label_encoder': label_encoder
}, 'product_classification_model_V1.joblib')
progress_bar.update(1)

# Complete progress bar
progress_bar.close()
print("Model training and saving complete.")
