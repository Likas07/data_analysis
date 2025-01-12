from autogluon.tabular import TabularPredictor, TabularDataset
import fireducks.pandas as pd 
import logging
import time
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Model paths
model_path = "/home/likas/workspace/github.com/Likas07/data_analysis/test/AutogluonModels/ag-20250109_234956"
backup_path = f"{model_path}_backup"

# Create backup
logger.info(f"Creating backup of model directory...")
logger.info(f"Source: {model_path}")
logger.info(f"Backup: {backup_path}")

if Path(backup_path).exists():
    logger.info("Backup already exists, skipping backup creation")
else:
    shutil.copytree(model_path, backup_path)
    logger.info("Backup created successfully")

# Load predictor
logger.info(f"\nLoading predictor from {model_path}")
predictor = TabularPredictor.load(model_path)

predictor.set_model_best(model='WeightedEnsemble_L2Best')

