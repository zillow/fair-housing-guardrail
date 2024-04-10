import os
import shutil
from io import StringIO

import pandas as pd

from fair_housing_guardrail.utils.fair_housing_classification import (
    FairHousingGuardrailClassification,
)
from fair_housing_guardrail.utils.helper import load_config, load_dataset, load_tokenizer

current_path = os.getcwd()

# Setup train
TRAIN_CONFIG_FILE_PATH = os.path.join(current_path, "examples/configs/train-config.yaml")
train_config = load_config(TRAIN_CONFIG_FILE_PATH)
train_config["output_data"]["output_dir"] = os.path.join(current_path, "tmp/")
train_config["output_data"]["model_output_dir"] = os.path.join(current_path, "tmp/model/")
train_config["input_data"]["stop_list_path"] = os.path.join(
    current_path, "examples/datasets/sample_stoplist.txt"
)
train_config["input_data"]["input_data_path"] = os.path.join(
    current_path, "examples/datasets/sample_data.jsonl"
)

# Setup test
TEST_CONFIG_FILE_PATH = os.path.join(current_path, "examples/configs/test-config.yaml")
test_config = load_config(TEST_CONFIG_FILE_PATH)
test_config["input_model"]["model_dir"] = os.path.join(current_path, "tmp/model/")
test_config["input_data"]["stop_list_path"] = os.path.join(
    current_path, "examples/datasets/sample_stoplist.txt"
)
test_config["input_data"]["input_data_path"] = os.path.join(
    current_path, "examples/datasets/sample_data.jsonl"
)
# Setup tokenizer
tokenizer = load_tokenizer()


def test_train_and_predict():
    with open(train_config["input_data"]["input_data_path"], "r") as file:
        json_data = file.read()
        train_dataset, test_dataset = load_dataset(train_config, tokenizer)

    instance = FairHousingGuardrailClassification(
        train_config, tokenizer, train_data=train_dataset, test_data=test_dataset
    )
    temp_dir = os.path.join(current_path, "tmp/")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Test train
    train_loss, eval_loss = instance.train()
    print(train_loss, eval_loss)

    instance.trainer.save_model()
    instance.model.save_pretrained(train_config["output_data"]["model_output_dir"])

    config_path = os.path.join(temp_dir, "config.json")

    assert os.path.exists(config_path)
    assert isinstance(train_loss, list)
    assert isinstance(eval_loss, list)
    assert len(train_loss) > 0
    assert len(eval_loss) > 0

    # Test predict
    with open(test_config["input_data"]["input_data_path"], "r") as file:
        json_data = file.read()
        predict_data = pd.read_json(StringIO(json_data), orient="records", lines=True)

    run_predict(predict_data)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def run_predict(predict_data):
    instance = FairHousingGuardrailClassification(test_config, tokenizer, None, predict_data)
    predictions = instance.predict()

    assert len(predictions) == 13
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], dict)
    assert "prediction" in predictions[0]
    assert "score" in predictions[0]
    assert "non-compliant-text" in predictions[0]
