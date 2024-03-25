# efficient_llm_data_wrangling

This github repo contains code for our submission: Towards Efficient Data Wrangling with LLMs using Code
Generation.

Following work by [Narayan et al.](https://arxiv.org/abs/2205.09911), we use the same set of benchmark [datasets](https://github.com/HazyResearch/fm_data_tasks).
You can clone the repo and download the data by using:

```
git clone git@github.com:effyli/efficient_llm_data_wrangling.git
mkdir data/
wget https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz -P data
tar xvf data/datasets.tar.gz -C data/
```

To run the script, first setup the data_dir environmental variable by using:

```
export DATASET_PATH="$PWD/data/datasets"
```

One example command of calling the script: 

```
python run_wrangler.py \
  --data_dir data/datasets/data_transformation/benchmark-bing-query-logs-semantics  \
  --num_trials 3  \
  --seed 42 \
  --k 3 \
  --d 0 \
  --num_iter 1
```

