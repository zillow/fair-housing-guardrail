import logging
import os

import evaluate
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fair_housing_guardrail.data.constants import get_label_mappings
from fair_housing_guardrail.data.json_dataset import JsonDataset
from fair_housing_guardrail.utils.stop_phrases import ProtectedAttributesStopWordsCheck

logger = logging.getLogger()
logger.setLevel(logging.INFO)

accuracy = evaluate.load("accuracy")
CONFIG_DICT = {}

global IS_BINARY


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if IS_BINARY:
        # Binary: sigmoid + threshold
        if "fairhousing" in CONFIG_DICT and "threshold" in CONFIG_DICT["fairhousing"]:
            threshold = CONFIG_DICT["fairhousing"]["threshold"]
        else:
            logger.info("Threshold not found in config, using 0.5 as threshold")
            threshold = 0.5
        probs = torch.sigmoid(torch.tensor(logits).squeeze(-1))
        preds = torch.where(
            probs > threshold,
            1,
            0,
        )
        return accuracy.compute(
            predictions=preds, references=torch.tensor(labels, dtype=torch.int64)
        )
    else:
        # Multi-class: argmax
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        return accuracy.compute(
            predictions=preds, references=torch.tensor(labels, dtype=torch.int64)
        )


def load_config(yaml_file_path):
    if not os.path.isfile(yaml_file_path):
        raise Exception("Config yaml file not found at path" + yaml_file_path)

    with open(yaml_file_path) as f:
        logger.info(f"Loading yaml config from path: {yaml_file_path}")
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(config)
        global CONFIG_DICT
        CONFIG_DICT = config
        global IS_BINARY
        IS_BINARY = config["input_data"]["is_binary"]
        return config


def load_tokenizer(config):
    if (
        "input_model_and_tokenizer" in config
        and config["input_model_and_tokenizer"]["model_and_tokenizer_dir"] is not None
    ):
        tokenizer_dir = config["input_model_and_tokenizer"]["model_and_tokenizer_dir"]
        print(f"Loading tokenizer from path: {tokenizer_dir}")
        return AutoTokenizer.from_pretrained(
            config["input_model_and_tokenizer"]["model_and_tokenizer_dir"]
        )

    if IS_BINARY:
        logger.info("Loading bert-base-uncased tokenizer")
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        logger.info("Loading roberta-large tokenizer")
        return AutoTokenizer.from_pretrained("roberta-large")


def load_model(config, num_labels):
    if "model" in config and config["model"]["do_train"] is True:
        if num_labels == 1:
            print("Train Mode: Loading bert-base-uncased")
            return AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels
            )
        else:
            print("Train Mode: Loading roberta-large")
            return AutoModelForSequenceClassification.from_pretrained(
                "roberta-large", num_labels=num_labels
            )

    if "input_model_and_tokenizer" not in config:
        raise Exception(
            "Error: must define either 'model' or 'input_model_and_tokenizer' in config."
        )
    if config["input_model_and_tokenizer"]["model_and_tokenizer_dir"] is None:
        raise Exception("Predict Mode: Unable to load model as model_dir is None")

    return AutoModelForSequenceClassification.from_pretrained(
        config["input_model_and_tokenizer"]["model_and_tokenizer_dir"], num_labels=num_labels
    )


def load_phrase_checker(stop_list_path):
    if stop_list_path is None:
        raise Exception("Stop list path is None")
    return ProtectedAttributesStopWordsCheck.get_instance(stop_list_path)


def load_dataset(config, tokenizer):
    pd_data = pd.read_json(config["input_data"]["input_data_path"], orient="records", lines=True)
    logger.info("Found the following columns in data: " + str(pd_data.columns))

    # Determine label mapping from the dataset
    labels = set(pd_data[config["input_data"]["label_column"]].unique())
    ID_TO_LABEL, LABEL_TO_ID = get_label_mappings(labels)

    if "model" in config and config["model"]["do_train"] is True:
        train_df, test_df = train_test_split(pd_data, test_size=0.1)
        logger.info("Splitting dataset into train and test sets")
        logger.info(f"Train count: {train_df.count()}")
        logger.info(f"Test count: {test_df.count()}")

        train_dataset = JsonDataset(train_df, LABEL_TO_ID, tokenizer, config)
        test_dataset = JsonDataset(test_df, LABEL_TO_ID, tokenizer, config)
    else:
        train_dataset = None
        test_dataset = pd_data
        logger.info(f"Test count: {pd_data.count()}")

    return train_dataset, test_dataset, LABEL_TO_ID, ID_TO_LABEL
