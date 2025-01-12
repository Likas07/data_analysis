from autogluon.tabular import TabularDataset, TabularPredictor
import fireducks.pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and prepare test data
test_data = pd.read_csv('Dimensions_not_registered.csv', delimiter=';', encoding='utf-8-sig')

# Load companies to disregard
disregard_companies = pd.read_csv('disregard_company.csv', delimiter=';', encoding='utf-8')
companies_to_filter = disregard_companies.iloc[:, 1].tolist()  # Get the second column (Company)

# Record initial size
initial_size = len(test_data)

# Filter out rows where owner matches any company in the disregard list
test_data = test_data[~test_data['Owner'].isin(companies_to_filter)]

# Log the filtering results
logger.info(f"Filtered out {initial_size - len(test_data)} rows where owner matched disregarded companies")
logger.info(f"Test dataset size after removing missing values: {len(test_data)} rows")
logger.info(f"Removed {len(test_data) - len(test_data)} rows with missing values")

test_data = TabularDataset(test_data)

# Initialize TabularPredictor
logger.info("Loading predictor model...")
predictor = TabularPredictor.load("/home/likas/workspace/github.com/Likas07/data_analysis/test/AutogluonModels/ag-20250106_131351", verbosity = 4)
# Time the prediction
logger.info("Starting prediction on test data...")
start_time = time.time()
y_pred = predictor.predict(test_data)
end_time = time.time()
prediction_time = end_time - start_time
logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
logger.info(f"Predictions: {y_pred.head()}")

logger.info("Saving predictions with confidence-based separation...")
start_time = time.time()

# Set your confidence threshold
CONFIDENCE_THRESHOLD = 0.90  # Adjust this value as needed

# Get predictions with probabilities
logger.info("Getting predictions with probabilities...")
start_proba_pred_time = time.time()
y_pred_proba = predictor.predict_proba(test_data)
end_proba_pred_time = time.time()
proba_pred_time = end_proba_pred_time - start_proba_pred_time
logger.info(f"Prediction with probabilities completed in {proba_pred_time:.2f} seconds")

# Create DataFrame with original data, predictions, and probabilities
results_df = test_data.copy()
results_df['Predicted_Ingredient'] = y_pred

# Add prediction confidence and top predictions
top_predictions = []
top_probabilities = []
max_probabilities = []

for idx in range(len(y_pred_proba)):
    probs = y_pred_proba.iloc[idx]
    top_3_idx = probs.nlargest(3).index
    top_3_pred = ', '.join(top_3_idx)
    top_3_prob = ', '.join([f'{probs[i]:.3f}' for i in top_3_idx])
    max_prob = probs.max()
    
    top_predictions.append(top_3_pred)
    top_probabilities.append(top_3_prob)
    max_probabilities.append(max_prob)

results_df['Top_3_Predictions'] = top_predictions
results_df['Top_3_Probabilities'] = top_probabilities
results_df['Confidence'] = max_probabilities

# Separate high and low confidence predictions
high_confidence = results_df[results_df['Confidence'] >= CONFIDENCE_THRESHOLD]
low_confidence = results_df[results_df['Confidence'] < CONFIDENCE_THRESHOLD]

# Save to separate Excel files
high_confidence.to_excel('predictions_high_confidence_ingredient.xlsx', index=False)
low_confidence.to_excel('predictions_for_review_ingredient.xlsx', index=False)

end_time = time.time()
logger.info(f"Results saved to separate files based on confidence threshold of {CONFIDENCE_THRESHOLD}")
logger.info(f"High confidence predictions: {len(high_confidence)} rows")
logger.info(f"Low confidence predictions requiring review: {len(low_confidence)} rows")
logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")

# Optional: Create a summary file with confidence distribution
confidence_summary = pd.DataFrame({
    'Confidence_Range': ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
    'Count': [
        len(results_df[results_df['Confidence'] < 0.2]),
        len(results_df[(results_df['Confidence'] >= 0.2) & (results_df['Confidence'] < 0.4)]),
        len(results_df[(results_df['Confidence'] >= 0.4) & (results_df['Confidence'] < 0.6)]),
        len(results_df[(results_df['Confidence'] >= 0.6) & (results_df['Confidence'] < 0.8)]),
        len(results_df[results_df['Confidence'] >= 0.8])
    ]
})
confidence_summary.to_excel('confidence_distribution_summary.xlsx', index=False)
