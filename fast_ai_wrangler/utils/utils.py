import logging
import re

from typing import List
from pathlib import Path
from rich.logging import RichHandler

# adapted from https://github.com/HazyResearch/fm_data_tasks/blob/updates/fm_data_tasks/utils/utils.py 
def setup_logger(log_dir: str):
    """Create log directory and logger."""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_dir) / "log.txt")
    handlers = [logging.FileHandler(log_path), RichHandler(rich_tracebacks=True)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )

def is_within_tolerance(predicted, actual, tolerance=0.021):
    """
    Check if the predicted value is within the tolerance of the actual value.

    Parameters:
    - predicted: The predicted value.
    - actual: The actual (ground truth) value.
    - tolerance: The tolerance level.

    Returns:
    - True if the difference is within the tolerance, False otherwise.
    """
    for pred, act in zip(predicted, actual):
        if  abs(pred - act) > tolerance:
            return False
    return True


def extract_numerical_values(s):
    """
    Extract all numerical values from a given string, including integers, floating point numbers, and fractions.

    Parameters:
    - s: The string containing numerical values.

    Returns:
    - A list of floats found in the string.
    """
    # This regex pattern matches integers, floating point numbers, and fractions.
    pattern = r"[-+]?([0-9]*\.?[0-9]+|[0-9]+\/[0-9]+)"
    matches = re.findall(pattern, s)
    numbers = []
    for match in matches:
        if '/' in match:
            # If the match is a fraction, split it and perform division to get the float equivalent.
            numerator, denominator = map(float, match.split('/'))
            number = numerator / denominator
        else:
            # Otherwise, directly convert the match to float.
            number = float(match)
        numbers.append(number)
    return numbers if numbers else None


def evaluate_numerical_values(pred, label):
    """
    Check if the prediction contains numerical values. 
    If yes, then check if difference between label and pred is within certain tolerance.
    """
    crc = None
    num_pred_try = extract_numerical_values(pred)
    num_gt_try = extract_numerical_values(label)
    if num_pred_try and num_gt_try:
        # num_label = extract_numerical_values(label)
        if is_within_tolerance(num_pred_try, num_gt_try):
            crc = True
        else:
            crc = False
    else:
        crc = False
    return crc
    

def compute_metrics(preds: List, golds: List, task: str):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    for pred, label in zip(preds, golds):
        if label:
            label = label.strip().lower()
        if isinstance(pred, tuple) or isinstance(pred, List):
            pred = ' '.join(str(element) for element in pred)
        elif pred == None:
            pred = ""
        if label == None:
            label = ""
        pred = pred.strip().lower()
        mets["total"] += 1
        if task in {
            "data_imputation",
            "entity_matching",
            "data_transformation",
            "error_detection_spelling"
        }:
            if pred == label:
                crc = True
            else:
                # if they are not equal, check if it is numerical values within tolerance
                crc = evaluate_numerical_values(pred, label)
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc:
                mets["tn"] += 1
            else:
                mets["fp"] += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1