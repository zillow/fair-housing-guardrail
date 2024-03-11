import os
import unittest.mock as mock
from io import StringIO

import pandas as pd

from fair_housing_guardrail.utils.fair_housing_classification import (
    FairHousingGuardrailClassification,
)
from fair_housing_guardrail.utils.helper import load_config, load_dataset, load_tokenizer

current_path = os.getcwd()
CONFIG_FILE_PATH = os.path.join(current_path, "examples/configs/test-config.yaml")
config = load_config(CONFIG_FILE_PATH)
config["input_model"]["model_dir"] = os.path.join(current_path, "examples/test_model/")
config["input_data"]["stop_list_path"] = os.path.join(
    current_path, "examples/datasets/sample_stoplist.txt"
)
config["input_data"]["input_data_path"] = os.path.join(
    current_path, "examples/datasets/sample_data.jsonl"
)
tokenizer = load_tokenizer()


@mock.patch("fair_housing_guardrail.utils.fair_housing_classification.SigmoidTrainer")
def test_train(mock_trainer_class):
    instance = FairHousingGuardrailClassification(config, tokenizer)

    mock_trainer = mock.Mock()
    mock_trainer.state.log_history = [
        {"loss": 0.1, "step": 1},
        {"eval_loss": 0.2, "step": 2},
        {"loss": 0.3, "step": 3},
        {"eval_loss": 0.4, "step": 4},
    ]
    mock_trainer_class.return_value = mock_trainer

    train_loss, eval_loss = instance.train()

    assert isinstance(train_loss, list)
    assert isinstance(eval_loss, list)

    assert len(train_loss) > 0
    assert len(eval_loss) > 0


with open(config["input_data"]["input_data_path"], "r") as file:
    json_data = file.read()
    test_data = pd.read_json(StringIO(json_data), orient="records", lines=True)


def test_predict():
    instance = FairHousingGuardrailClassification(config, tokenizer, None, test_data)
    predictions = instance.predict()

    assert len(predictions) == 13
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], dict)
    assert "prediction" in predictions[0]
    assert "score" in predictions[0]
    assert "non-compliant-text" in predictions[0]
