import json
import logging
import argparse
import numpy as np
import pandas as pd

from typing import List
from pathlib import Path
from utils.utils import setup_logger, compute_metrics
import utils.data_utils as data_utils
from function_generator import generate_function_pipeline, evaluate


logger = logging.getLogger(__name__)

def function_generation(args, train_data, instruction, task):
    """
    Given training data, we sample demonstrations and generate functions by calling LLMs.
    """
    # get samples used for prompt from train data
    saved_funcs = []
    saved_acc = []
    saved_trials = []
    for trial_num in range(args.num_trials):
        seed = args.seed + trial_num
        np.random.seed(seed)
        if task in ["entity_matching", "error_detection_spelling"]:
            demonstrations = data_utils.sample_data_stratified(train_data, args.k)
        elif task in ["data_transformation", "data_imputation"]:
            demonstrations = data_utils.sample_data_random(train_data, args.k)
        logger.info(demonstrations)
        if len(train_data) > args.d:
            supervision_data_list = data_utils.sample_data_random(train_data, args.d)
        else:
            supervision_data_list = None
        func, acc = generate_function_pipeline(instruction, demonstrations, task=task, supervision_data=supervision_data_list)
        num_iter = 1
        while "No function" in func:
            if num_iter >= args.num_iter:
                # use dummy function string
                logger.info("Using dummy function.")
                func = "def string_transformation(input_string): return None"
                continue
            seed += 1234
            np.random.seed(seed)
            demonstrations = data_utils.sample_data_random(train_data, args.k)
            func, acc = generate_function_pipeline(instruction, demonstrations, task=task, supervision_data=supervision_data_list)
            num_iter += 1
        # now we calculate metrics
        # run the function on test set
        logger.info("Learned function for trial {} is {}".format(trial_num, func))
        saved_funcs.append(func)
        saved_acc.append(acc)
        # not every trial might have working function
        saved_trials.append(trial_num)
    return saved_funcs, saved_acc, saved_trials


def main():
    # get arguments
    parser = argparse.ArgumentParser(description="Run wrangler")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory.", default="outputs"
    )
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=3)
    parser.add_argument("--d", type=int, help="Number examples for training a classifier", default=100)
    parser.add_argument("--num_iter", type=int, help="Number of iterations to sample from training data", default=1)
    parser.add_argument(
        "--num_run",
        type=int,
        help="Number examples to run through model.",
        default=-1,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number trials to run. Results will be averaged with variance reported.",
        default=1,
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="random",
        choices=["random", "manual", "validation_clusters"],
    )
    parser.add_argument(
        "--class_balanced",
        help="Class balance training data. Good for classification tasks \
             with random prompts.",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '^'.",
        default="^",
    )
    parser.add_argument(
        "--nan_tok",
        type=str,
        help="Token to represent nan entries. Default is 'nan'.",
        default="nan",
    )
    args = parser.parse_args()
    # Get absolute path
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))

    test_file = "test"

    # Read pandas DF datasets
    pd_data_files, task = data_utils.read_data(
        data_dir=args.data_dir,
        class_balanced=args.class_balanced,
        add_instruction=False,
        max_train_samples=-1,
        max_train_percent=-1,
        sep_tok=args.sep_tok,
        nan_tok=args.nan_tok,
    )
    logger.info(f"Task for the dataset is {task}")
    if test_file not in pd_data_files:
        raise ValueError(f"Need {test_file} data")

    train_data = pd_data_files["train"]
    test_data = pd_data_files[test_file]
    num_run = args.num_run
    instructions = []
    if isinstance(train_data, List):
        if args.num_run == -1:
            num_run = test_data[0].shape[0]
        num_run = min(num_run, test_data[0].shape[0])
        logger.info(f"Number of tasks is {len(train_data)}")
        logger.info(f"Train shape is {train_data[0].shape[0]}")
        logger.info(f"Test shape is {test_data[0].shape[0]}")
        logger.info(f"Running {num_run} examples for {args.num_trials} trials.")
        if "instructions" in pd_data_files:
            instructions = pd_data_files["instructions"]
        logger.info(f"Number of tasks is {len(instructions)}")

    elif isinstance(train_data, pd.DataFrame):
        if args.num_run == -1:
            num_run = test_data.shape[0]
        num_run = min(num_run, test_data.shape[0])
        logger.info(f"Train shape is {train_data.shape[0]}")
        logger.info(f"Test shape is {test_data.shape[0]}")
        logger.info(f"Running {num_run} examples for {args.num_trials} trials.")
        train_data = [train_data]
        test_data = [test_data]
        instruction = data_utils.read_instruction(args.data_dir)
        if instruction:
            instructions = [instruction]
        else:
            instructions = [""]

    task_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}
    all_gts = []
    all_preds = []
    task_number = 0
    for train_data_pd, test_data_pd, instruction in zip(train_data, test_data, instructions):
        task_number += 1
        logger.info(f"Task instruction {instruction}")
        if len(test_data_pd) == 0:
            logger.info("Not enough samples to run, continue to next task.")
            continue
        saved_funcs = []
        # we have a list of dataframes where each dataframe represent a task 
        saved_funcs, saved_acc, saved_trials = function_generation(args, train_data_pd, instruction, task=task)
        if not saved_funcs:
            print("No functions found, change your seed and run again!")
            exit()
        
        batches = []
        test_data_lst = data_utils.deserialize_data(test_data_pd)
        batches = [test_data_lst]
        
        accs_per_batch = []
        gts_per_task = []
        preds_per_task = []
        i = 0
        logger.info(f"number of batches splitted is {len(batches)}")
        logger.info(f"number of funcs is {len(saved_funcs)}")

        # evaluate per batch
        if task in ["data_transformation", "error_detection_spelling"]:
            for batch, func in zip(batches, saved_funcs):
                logger.info(f"using function {func} to evaluate")
                acc, preds = evaluate(func, batch, task)
                logger.info(f"acc: {acc} for batch {i}")
                accs_per_batch.append(acc)
                for sample, pred in zip(batch, preds):
                    gt = sample["Output"]
                    gts_per_task.append(gt)
                    preds_per_task.append(pred)
                    all_gts.append(gt)
                    all_preds.append(pred)
                    logger.info(f"====> pred: {pred} <====")
                    logger.info(f"====> gt: {gt} <====")
                i += 1
        else:
            sorted_acc, sorted_funcs = zip(*sorted(zip(saved_acc, saved_funcs), key=lambda x: x[0], reverse=True))
            func = sorted_funcs[0]
            logger.info(f"sorted acc {sorted_acc}, sorted funcs {sorted_funcs}")
            logger.info(f"Acc for the fincal func is {sorted_acc[0]}, Final function to use is {func}")           
            for batch in batches:
                
                acc, preds = evaluate(func, batch, task)
                logger.info(f"acc: {acc} for batch {i}")
                
                accs_per_batch.append(acc)
                for sample, pred in zip(batch, preds):
                    gt = sample["Output"]
                    gts_per_task.append(gt)
                    preds_per_task.append(pred)
                    all_gts.append(gt)
                    all_preds.append(pred)
                    logger.info(f"====> pred: {pred} <====")
                    logger.info(f"====> gt: {gt} <====")
        
        # calculating metrics per task
        prec, rec, acc, f1 = compute_metrics(preds_per_task, gts_per_task, task)
        logger.info(f"Task number {task_number}")
        logger.info(
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
    
        task_metrics["rec"].append(rec)
        task_metrics["prec"].append(prec)
        task_metrics["acc"].append(acc)
        task_metrics["f1"].append(f1)


    output_file = (
        Path(args.output_dir)
        / f"{Path(args.data_dir).stem}"
        / f"{test_file}"
        / f"{args.k}k"
        / f"{args.d}d"
        f"_{int(args.class_balanced)}cb"
        f"_{args.sample_method}"
        f"_{args.num_run}run" / f"trial_{num_run}.feather"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved to {output_file}")
    
    # calculate metrics all rows
    prec, rec, acc, f1 = compute_metrics(all_preds, all_gts, task=task)
    task_metrics["prec_on_rows"] = prec
    task_metrics["rec_on_rows"] = rec
    task_metrics["acc_on_rows"] = acc
    task_metrics["f1_on_rows"] = f1

    for k, values in list(task_metrics.items()):
        task_metrics[f"{k}_avg"] = np.average(values)
        task_metrics[f"{k}_std"] = np.std(values)

    output_metrics = output_file.parent / "metrics.json"
    json.dump(task_metrics, open(output_metrics, "w"))

    output_functions = output_file.parent / "learned_funcs.json"
    json.dump(saved_funcs, open(output_functions, "w"))
    logger.info(f"Final Metrics {json.dumps(task_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")
    logger.info(f"Learned funcs dumped to {output_functions}")



if __name__ == "__main__":
    main()