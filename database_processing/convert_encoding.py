import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_file_encoding(input_file, output_file=None):
    """
    Convert a CSV file from ISO-8859-1 to UTF-8 with BOM
    """
    if output_file is None:
        output_file = input_file
    
    try:
        # Read the file with original encoding
        logger.info(f"Reading file {input_file} with ISO-8859-1 encoding...")
        df = pd.read_csv(input_file, delimiter=';', encoding='iso-8859-1')
        
        # Save with new encoding
        logger.info(f"Saving file to {output_file} with UTF-8-SIG encoding...")
        df.to_csv(output_file, sep=';', encoding='utf-8-sig', index=False)
        logger.info("Conversion completed successfully!")
        
        # Verify the conversion
        logger.info("Verifying the converted file...")
        df_verify = pd.read_csv(output_file, delimiter=';', encoding='utf-8-sig')
        logger.info(f"Verification successful! File has {len(df_verify)} rows and {len(df_verify.columns)} columns")
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = 'train_trademark.csv'
    # Create a backup of the original file
    backup_file = 'train_trademark_backup.csv'
    
    try:
        # First create a backup
        logger.info(f"Creating backup as {backup_file}...")
        df_backup = pd.read_csv(input_file, delimiter=';', encoding='iso-8859-1')
        df_backup.to_csv(backup_file, sep=';', encoding='iso-8859-1', index=False)
        logger.info("Backup created successfully")
        
        # Convert the original file
        convert_file_encoding(input_file)
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        logger.info("You can find the original file in the backup: train_trademark_backup.csv")
