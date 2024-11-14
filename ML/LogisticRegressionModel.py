import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
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
filtered_data = filtered_data.merge(
    ft_ingredients_data[['INGREDIENT_NAME_IMPORT', 'OR_INGREDIENT_LABEL']],
    left_on='Ingredient_Match', right_on='INGREDIENT_NAME_IMPORT', how='left'
).drop(columns=['Ingredient_Match', 'INGREDIENT_NAME_IMPORT'])
progress_bar.update(1)
del dimension_batches, ingredient_matches  # Free memory
gc.collect()

# Step 6: Parallel Text Cleaning and Combination
logger.info("Cleaning and combining text fields...")

def clean_text(text):
    text = unidecode(text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

with ProcessPoolExecutor() as executor:
    filtered_data['cleaned_description'] = list(executor.map(
        clean_text,
        filtered_data['Dimension Value'] + ' ' + filtered_data['Owner'] + ' ' + filtered_data['NAM_TRADEMARK_IMPORT'].fillna('') + ' ' + filtered_data['OR_INGREDIENT_LABEL'].fillna('')
    ))
progress_bar.update(1)

# Step 7: Vectorization and Model Training

logger.info("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), stop_words='english')
X_tfidf = vectorizer.fit_transform(filtered_data['cleaned_description'])
X_tfidf = csr_matrix(X_tfidf)  # Convert to sparse format
X_combined = hstack([X_tfidf, filtered_data[['Type_Label']].values])

# Encode labels
logger.info("Encoding target labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(filtered_data['NAM_TRADEMARK_IMPORT'].fillna('UNKNOWN'))
progress_bar.update(1)

# After combining the features, ensure the sparse matrix is in CSR format for slicing
X_combined = hstack([X_tfidf, csr_matrix(filtered_data[['Type_Label']].values)])
X_combined = X_combined.tocsr()  # Convert to CSR format for slicing

# Step 8: Train with SGDClassifier for efficiency
logger.info("Training logistic regression model using SGDClassifier...")
model = SGDClassifier(loss="log_loss", max_iter=1000)

# Dynamically adjust train_batch_size if larger than dataset size
train_batch_size = min(500, X_combined.shape[0])
if X_combined.shape[0] > 0:
    for start in range(0, X_combined.shape[0], train_batch_size):
        end = start + train_batch_size
        X_batch = X_combined[start:end]  # Slicing is now possible in CSR format
        y_batch = y[start:end]
        model.partial_fit(X_batch, y_batch, classes=np.unique(y))
progress_bar.update(1)


# Step 9: Evaluate the model
logger.info("Evaluating the model...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Ensure only labels that appear in y_test are used
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    labels=np.unique(y_test),  # Only use labels present in y_test
    target_names=label_encoder.inverse_transform(np.unique(y_test))
))
progress_bar.update(1)


# Step 10: Save the model, vectorizer, and label encoder
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
