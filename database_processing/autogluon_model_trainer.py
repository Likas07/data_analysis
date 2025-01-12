from autogluon.tabular import TabularDataset, TabularPredictor
import fireducks.pandas as pd
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Read CSV with correct delimiter and encoding for train_trademark.csv (UTF-8 with BOM)
try:
    df = pd.read_csv('train_trademark.csv', delimiter=';', encoding='utf-8-sig')
    logger.info("Successfully read training data")
except Exception as e:
    logger.error(f"Error reading training data: {str(e)}")
    raise

logger.info(f"Original column names: {df.columns.tolist()}")
logger.info(f"Original dataset size: {len(df)} rows")

# Remove rows with missing values in the Trademark column
df_clean = df.dropna(subset=['Trademark'])
logger.info(f"Dataset size after removing missing values: {len(df_clean)} rows")
logger.info(f"Removed {len(df) - len(df_clean)} rows with missing values")

# Convert to TabularDataset
train_data = TabularDataset(df_clean)
logger.info("\nAnalyzing Trademark column (index 2):")
logger.info(f"Column statistics: {train_data[2].describe()}")

# Load and prepare test data
try:
    test_data = pd.read_csv('Dimensions_not_registered.csv', delimiter=';', encoding='utf-8-sig')
    logger.info("Successfully read test data")
except Exception as e:
    logger.error(f"Error reading test data: {str(e)}")
    raise

# Load companies to disregard
try:
    disregard_companies = pd.read_csv('disregard_company.csv', delimiter=';', encoding='utf-8-sig')
    logger.info("Successfully read disregard companies data")
except Exception as e:
    logger.error(f"Error reading disregard companies data: {str(e)}")
    raise

# Record initial size
initial_size = len(test_data)

# Filter out rows where owner matches any company in the disregard list
test_data = test_data[~test_data['Owner'].isin(disregard_companies.iloc[:, 1].tolist())]

# Log the filtering results
logger.info(f"Filtered out {initial_size - len(test_data)} rows where owner matched disregarded companies")
logger.info(f"Test dataset size: {len(test_data)} rows")

test_data = TabularDataset(test_data)

# Initialize TabularPredictor with timing
logger.info("Starting model training...")
start_time = time.time()
predictor = TabularPredictor(label=2, eval_metric='accuracy').fit(train_data, time_limit=14400, infer_limit=0.05, presets='best', excluded_model_types=['CAT'])
training_time = time.time() - start_time
logger.info(f"Model training completed in {training_time:.2f} seconds")

# Make predictions
logger.info("Making predictions on test data...")
start_time = time.time()
y_pred = predictor.predict(test_data)
prediction_time = time.time() - start_time
logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
logger.info(f"First few predictions: {y_pred.head()}")

# Evaluate model
logger.info("Evaluating model performance...")
evaluation = predictor.evaluate(test_data, silent=True)
logger.info(f"Model evaluation results: {evaluation}")

# Show leaderboard
logger.info("Generating model leaderboard...")
leaderboard = predictor.leaderboard(test_data, silent=True)
logger.info(f"Model leaderboard:\n{leaderboard}")

# Save model
save_path = 'autogluon_model_trademark'
logger.info(f"Saving model to {save_path}...")
predictor.save(save_path)
logger.info("Model saved successfully")
