########################################
# adapted from: https://github.com/HazyResearch/fm_data_tasks/blob/updates/fm_data_tasks/utils/data_utils.py 
########################################
"""Constants."""

import os
from pathlib import Path

DATASET_PATH = os.environ.get("DATASET_PATH", Path("data/datasets").resolve())

DATA2TASK = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "entity_matching",
    f"{DATASET_PATH}/data_imputation/Buy": "data_imputation",
    f"{DATASET_PATH}/data_imputation/Restaurant": "data_imputation",
    f"{DATASET_PATH}/error_detection/Hospital": "error_detection_spelling",
    f"{DATASET_PATH}/error_detection/Adult": "error_detection_spelling",
    f"{DATASET_PATH}/schema_matching/Synthea": "schema_matching",
    f"{DATASET_PATH}/data_transformation/benchmark-bing-query-logs": "data_transformation",
    f"{DATASET_PATH}/data_transformation/benchmark-stackoverflow": "data_transformation",
    f"{DATASET_PATH}/data_transformation/benchmark-FF-Trifacta-GoogleRefine": "data_transformation",
    f"{DATASET_PATH}/data_transformation/benchmark-headcase": "data_transformation",
    f"{DATASET_PATH}/data_transformation/benchmark-bing-query-logs-semantics": "data_transformation",
    f"{DATASET_PATH}/data_transformation/benchmark-bing-query-logs-unit": "data_transformation"
}

IMPUTE_COLS = {
    f"{DATASET_PATH}/data_imputation/Buy": "manufacturer",
    f"{DATASET_PATH}/data_imputation/Restaurant": "city",
}


