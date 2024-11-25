import cudf
import cupy as cp
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.ensemble import RandomForestClassifier as cuRFClassifier
from cuml.preprocessing import LabelEncoder as cuLabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
import faiss
import torch
from tqdm import tqdm
import optuna
import logging
import os
import joblib
import gc

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BERT_TEXT_LEN = 512
BATCH_SIZE = 1000  # Batch size for processing
os.makedirs("intermediate_data", exist_ok=True)  # Directory for saving intermediate data


def load_data():
    logger.info("Loading data...")

    # Use pandas to load Excel data
    trademark_data = pd.read_excel('Dimensions not registered -2024-10-24-08-00-26.xlsx', sheet_name='TRADEMARK')
    ft_trademark_data = pd.read_excel('tb_ft_trademark.xlsx')
    ft_ingredients_data = pd.read_excel('tb_ft_ingredients.xlsx')
    disregard_companies = pd.read_excel('disregard_company.xlsx', sheet_name='DISREGARD')

    # Convert pandas DataFrame to cuDF DataFrame
    trademark_data = cudf.DataFrame(trademark_data)
    ft_trademark_data = cudf.DataFrame(ft_trademark_data)
    ft_ingredients_data = cudf.DataFrame(ft_ingredients_data)
    disregard_companies = cudf.DataFrame(disregard_companies)

    disregard_list = disregard_companies['Company'].dropna().to_pandas().tolist()
    return trademark_data, ft_trademark_data, ft_ingredients_data, disregard_list


# Step 2: Data Filtering
def filter_data(trademark_data, disregard_list):
    logger.info("Filtering data...")
    return trademark_data[
        (trademark_data['DataSourceName'] == "Base Brasil - Logcomex") &
        (~trademark_data['Owner'].isin(disregard_list))
    ].reset_index(drop=True)


# Step 3: Text Cleaning
def clean_text_gpu(text_col):
    text_col = text_col.str.lower()
    text_col = text_col.str.replace(r'[^\w\s]', '', regex=True)
    text_col = text_col.str.strip()
    return text_col


# Step 4: Build FAISS Index
def build_faiss_index(vectors, dimension):
    logger.info("Building FAISS index...")
    nlist = 100
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, f"intermediate_data/faiss_index_{dimension}.index")
    return index


# Step 5: Generate BERT Embeddings
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, tokenizer):
        self.text_data = text_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_BERT_TEXT_LEN)
        return tokens


def get_bert_embeddings(data, tokenizer, model):
    dataset = BERTDataset(data, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating BERT Embeddings"):
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    np.save("intermediate_data/bert_embeddings.npy", embeddings)  # Save embeddings
    return embeddings


# Step 6: Optuna Hyperparameter Optimization
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 10, 50)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Define model
    model = cuRFClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42
    )

    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy  # Optuna will maximize accuracy


def perform_optuna_optimization():
    logger.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best accuracy: {study.best_value}")
    return study.best_params


# Main Pipeline
def main():
    # Load data
    trademark_data, ft_trademark_data, ft_ingredients_data, disregard_list = load_data()

    # Filter and preprocess data
    filtered_data = filter_data(trademark_data, disregard_list)

    # Text cleaning
    filtered_data['Owner'] = clean_text_gpu(filtered_data['Owner'])
    filtered_data['Dimension Value'] = clean_text_gpu(filtered_data['Dimension Value'])

    # TF-IDF Vectorization
    logger.info("Vectorizing data with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectors = tfidf_vectorizer.fit_transform(filtered_data['Dimension Value'].to_pandas())

    # Build FAISS index
    faiss_index = build_faiss_index(tfidf_vectors, tfidf_vectors.shape[1])

    # BERT Embeddings
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-multilingual-cased')
    bert_model = DistilBertModel.from_pretrained('distilbert-multilingual-cased').to(device)
    bert_embeddings = get_bert_embeddings(filtered_data['Dimension Value'].to_pandas().tolist(), tokenizer, bert_model)

    # Prepare combined feature matrix and labels
    global X_combined, y  # Needed for Optuna's objective
    X_combined = cp.hstack([tfidf_vectors, cp.array(filtered_data[['Type_Label']].values), bert_embeddings])
    label_encoder = cuLabelEncoder()
    y = label_encoder.fit_transform(filtered_data['Owner'].fillna('UNKNOWN').to_pandas())

    # Hyperparameter Optimization
    best_params = perform_optuna_optimization()

    # Final Model Training
    logger.info("Training the final model with best parameters...")
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    final_model = cuRFClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluate final model
    y_pred = final_model.predict(X_test)
    logger.info("Final Model Evaluation:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(classification_report(y_test, y_pred))

    # Save final model
    joblib.dump(final_model, "intermediate_data/final_model.joblib")
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    main()
