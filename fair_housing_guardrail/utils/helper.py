import logging
import os

import evaluate
import pandas as pd
import torch
import yaml
from data.constants import LABEL_TO_ID
from data.json_dataset import JsonDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.stop_phrases import ProtectedAttributesStopWordsCheck

logger = logging.getLogger()
logger.setLevel(logging.INFO)

accuracy = evaluate.load("accuracy")
CONFIG_DICT = {}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if "fairhousing" in CONFIG_DICT and "threshold" in CONFIG_DICT["fairhousing"]:
        threshold = CONFIG_DICT["fairhousing"]["threshold"]
    else:
        logger.info("Threshold not found in config, using 0.5 as threshold")
        threshold = 0.5
    preds = torch.where(
        torch.nn.functional.sigmoid(torch.tensor(predictions)).reshape(
            -1,
        )
        > threshold,
        1,
        0,
    )
    return accuracy.compute(predictions=preds, references=torch.tensor(labels, dtype=torch.int32))


def load_config(yaml_file_path):
    if not os.path.isfile(yaml_file_path):
        raise Exception("Config yaml file not found at path" + yaml_file_path)

    with open(yaml_file_path) as f:
        logger.info(f"Loading yaml config from path: {yaml_file_path}")
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(config)
        global CONFIG_DICT
        CONFIG_DICT = config
        return config


def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def load_model(config):
    if "model" in config and config["model"]["do_train"] is True:
        logger.info("Train Mode: Loading bert-base-uncased")
        return AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=1
        )
    if "input_model" not in config:
        raise Exception("Error: must define either 'model' or 'input_model' in config.")
    if config["input_model"]["model_dir"] is None:
        raise Exception("Predict Mode: Unable to load model as model_dir is None")

    return AutoModelForSequenceClassification.from_pretrained(
        config["input_model"]["model_dir"], num_labels=1
    )


def load_phrase_checker(stop_list_path):
    if stop_list_path is None:
        raise Exception("Stop list path is None")
    return ProtectedAttributesStopWordsCheck.get_instance(stop_list_path)


def load_dataset(config, tokenizer):
    pd_data = pd.read_json(config["input_data"]["input_data_path"], orient="records", lines=True)
    logger.info("Found the following columns in data: " + pd_data.columns)

    if "model" in config and config["model"]["do_train"] is True:
        train_df, test_df = train_test_split(pd_data, test_size=0.5)
        logger.info("Splitting dataset into train and test sets")
        logger.info(f"Train count: {train_df.count()}")
        logger.info(f"Test count: {test_df.count()}")
        train_dataset = JsonDataset(train_df, LABEL_TO_ID, tokenizer, config)
        test_dataset = JsonDataset(test_df, LABEL_TO_ID, tokenizer, config)
    else:
        train_dataset = None
        test_dataset = pd_data
        logger.info(f"Test count: {pd_data.count()}")

    return train_dataset, test_dataset
