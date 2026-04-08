import logging
import os
import shutil
from io import StringIO

import pandas as pd

from fair_housing_guardrail.utils.fair_housing_classification import (
    FairHousingGuardrailClassification,
)
from fair_housing_guardrail.utils.helper import (
    load_config,
    load_dataset,
    load_tokenizer,
)

current_path = os.getcwd()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Use file names from examples/configs and examples/datasets
BINARY_TRAIN_CONFIG_FILE_PATH = os.path.join(
    current_path, "examples/configs/train-config-binary.yaml"
)
BINARY_TEST_CONFIG_FILE_PATH = os.path.join(
    current_path, "examples/configs/test-config-binary.yaml"
)
MULTICLASS_TRAIN_CONFIG_FILE_PATH = os.path.join(
    current_path, "examples/configs/train-config-multiclass.yaml"
)
MULTICLASS_TEST_CONFIG_FILE_PATH = os.path.join(
    current_path, "examples/configs/test-config-multiclass.yaml"
)

BINARY_SAMPLE_DATA_PATH = os.path.join(current_path, "examples/datasets/sample_data_binary.jsonl")
MULTICLASS_SAMPLE_DATA_PATH = os.path.join(
    current_path, "examples/datasets/sample_data_multiclass.jsonl"
)
STOPLIST_PATH = os.path.join(current_path, "examples/datasets/sample_stoplist.txt")


def test_train_and_predict_binary():
    binary_train_config = load_config(BINARY_TRAIN_CONFIG_FILE_PATH)
    binary_train_config["output_data"]["output_dir"] = os.path.join(current_path, "tmp/")
    binary_train_config["output_data"]["model_output_dir"] = os.path.join(
        current_path, "tmp/binary_model/"
    )
    binary_train_config["input_data"]["stop_list_path"] = STOPLIST_PATH
    binary_train_config["input_data"]["input_data_path"] = BINARY_SAMPLE_DATA_PATH
    binary_tokenizer = load_tokenizer(binary_train_config)
    train_dataset, test_dataset, label_to_id, id_to_label = load_dataset(
        binary_train_config, binary_tokenizer
    )

    instance = FairHousingGuardrailClassification(
        binary_train_config,
        tokenizer=binary_tokenizer,
        train_data=train_dataset,
        test_data=test_dataset,
        LABEL_TO_ID=label_to_id,
        ID_TO_LABEL=id_to_label,
    )

    temp_dir = os.path.join(current_path, "tmp/")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Test train
    train_loss, eval_loss = instance.train()
    print(train_loss, eval_loss)

    instance.trainer.save_model()
    instance.model.save_pretrained(binary_train_config["output_data"]["model_output_dir"])

    config_path = os.path.join(temp_dir, "config.json")

    assert os.path.exists(config_path)
    assert isinstance(train_loss, list)
    assert isinstance(eval_loss, list)
    assert len(train_loss) > 0
    assert len(eval_loss) > 0

    run_predict_binary(binary_tokenizer, label_to_id, id_to_label)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def run_predict_binary(binary_tokenizer, label_to_id, id_to_label):
    binary_test_config = load_config(BINARY_TEST_CONFIG_FILE_PATH)
    binary_test_config["input_model_and_tokenizer"]["model_and_tokenizer_dir"] = os.path.join(
        current_path, "tmp/binary_model/"
    )
    binary_test_config["input_data"]["stop_list_path"] = STOPLIST_PATH
    binary_test_config["input_data"]["input_data_path"] = BINARY_SAMPLE_DATA_PATH

    # Test predict
    with open(binary_test_config["input_data"]["input_data_path"], "r") as file:
        json_data = file.read()
        predict_data = pd.read_json(StringIO(json_data), orient="records", lines=True)

    instance = FairHousingGuardrailClassification(
        binary_test_config,
        tokenizer=binary_tokenizer,
        train_data=None,
        test_data=predict_data,
        LABEL_TO_ID=label_to_id,
        ID_TO_LABEL=id_to_label,
    )

    predictions = instance.predict()

    assert len(predictions) == 13
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], dict)
    assert "prediction" in predictions[0]
    assert "score" in predictions[0]
    assert "non-compliant-text" in predictions[0]


def test_train_and_predict_multiclass():
    multiclass_train_config = load_config(MULTICLASS_TRAIN_CONFIG_FILE_PATH)
    multiclass_train_config["output_data"]["output_dir"] = os.path.join(current_path, "tmp/")
    multiclass_train_config["output_data"]["model_output_dir"] = os.path.join(
        current_path, "tmp/multiclass_model/"
    )
    multiclass_train_config["input_data"]["stop_list_path"] = STOPLIST_PATH
    multiclass_train_config["input_data"]["input_data_path"] = MULTICLASS_SAMPLE_DATA_PATH

    multiclass_tokenizer = load_tokenizer(multiclass_train_config)
    train_dataset, test_dataset, label_to_id, id_to_label = load_dataset(
        multiclass_train_config, multiclass_tokenizer
    )

    instance = FairHousingGuardrailClassification(
        multiclass_train_config,
        tokenizer=multiclass_tokenizer,
        train_data=train_dataset,
        test_data=test_dataset,
        LABEL_TO_ID=label_to_id,
        ID_TO_LABEL=id_to_label,
    )
    temp_dir = os.path.join(current_path, "tmp/")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Test train
    train_loss, eval_loss = instance.train()
    print(train_loss, eval_loss)

    instance.trainer.save_model()
    instance.model.save_pretrained(multiclass_train_config["output_data"]["model_output_dir"])

    config_path = os.path.join(temp_dir, "config.json")

    assert os.path.exists(config_path)
    assert isinstance(train_loss, list)
    assert isinstance(eval_loss, list)
    assert len(train_loss) > 0
    assert len(eval_loss) > 0

    run_predict_multiclass(multiclass_tokenizer, label_to_id, id_to_label)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def run_predict_multiclass(multiclass_tokenizer, label_to_id, id_to_label):
    multiclass_test_config = load_config(MULTICLASS_TEST_CONFIG_FILE_PATH)
    multiclass_test_config["input_model_and_tokenizer"]["model_and_tokenizer_dir"] = os.path.join(
        current_path, "tmp/multiclass_model/"
    )
    multiclass_test_config["input_data"]["stop_list_path"] = STOPLIST_PATH
    multiclass_test_config["input_data"]["input_data_path"] = MULTICLASS_SAMPLE_DATA_PATH

    # Test predict
    with open(multiclass_test_config["input_data"]["input_data_path"], "r") as file:
        json_data = file.read()
        predict_data = pd.read_json(StringIO(json_data), orient="records", lines=True)

    instance = FairHousingGuardrailClassification(
        multiclass_test_config,
        tokenizer=multiclass_tokenizer,
        train_data=None,
        test_data=predict_data,
        LABEL_TO_ID=label_to_id,
        ID_TO_LABEL=id_to_label,
    )
    predictions = instance.predict()

    assert len(predictions) == 16
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], dict)
    assert "prediction" in predictions[0]
    assert "score" in predictions[0]
    assert "non-compliant-text" in predictions[0]
