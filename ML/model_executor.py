import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode
import re

# Load the saved model, vectorizer, and label encoder
model_data = joblib.load('product_classification_model_V1.joblib')
model = model_data['model']
vectorizer = model_data['vectorizer']
label_encoder = model_data['label_encoder']

# Load the test data (replace 'Dimensions not registered -2024-10-24-08-00-26.xlsx' with your test file if different)
test_data = pd.read_excel('Dimensions not registered -2024-10-24-08-00-26.xlsx', sheet_name='TRADEMARK', usecols=['Dimension Value', 'Owner', 'NCM'])

# Preprocessing function
def preprocess_text(text):
    text = unidecode(text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

# Step 1: Clean the text in Dimension Value and Owner columns and combine them
test_data['cleaned_description'] = (test_data['Dimension Value'] + ' ' + test_data['Owner']).fillna('').apply(preprocess_text)

# Step 2: Vectorize the cleaned descriptions
X_tfidf_test = vectorizer.transform(test_data['cleaned_description'])

# Step 3: Create Type_Label based on the NCM column
def label_technical_formulated(ncm):
    if str(ncm).startswith("2"):
        return 0  # TECHNICAL
    elif str(ncm).startswith("3"):
        return 1  # FORMULATED
    else:
        return -1  # UNKNOWN or other

test_data['Type_Label'] = test_data['NCM'].apply(label_technical_formulated)

# Step 4: Combine the vectorized data with Type_Label feature
X_combined_test = hstack([X_tfidf_test, test_data[['Type_Label']].values])

# Step 5: Predict the labels using the loaded model
predictions = model.predict(X_combined_test)
predicted_labels = label_encoder.inverse_transform(predictions)

# Step 6: Create output DataFrame with required columns
output_df = pd.DataFrame({
    'Dimension Value': test_data['Dimension Value'],
    'Predicted Label': predicted_labels,
    'Owner': test_data['Owner'],
    'TECHNICAL or FORMULATED': test_data['Type_Label'].apply(lambda x: 'TECHNICAL' if x == 0 else 'FORMULATED' if x == 1 else 'UNKNOWN')
})

# Step 7: Save the results to an Excel file
output_df.to_excel('model_predictions_output.xlsx', index=False)
print("Predictions saved to 'model_predictions_output.xlsx'")
