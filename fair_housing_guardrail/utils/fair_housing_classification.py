import logging
from typing import Dict, List, Tuple, Union

import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from fair_housing_guardrail.data.constants import ID_TO_LABEL
from fair_housing_guardrail.utils.helper import (
    compute_metrics,
    load_model,
    load_phrase_checker,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FairHousingGuardrailClassification:
    def __init__(self, config={}, tokenizer=None, train_data=None, test_data=None):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.config)
        self.model = self.model.to(self.device)
        self.phrase_checker = load_phrase_checker(self.config["input_data"]["stop_list_path"])
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.predictions = []

    def train(self) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        # Set training arguments from config if present otherwise set default values
        config_model = self.config.get("model", {})
        training_args = TrainingArguments(
            output_dir=self.config["output_data"]["output_dir"],
            learning_rate=float(config_model.get("learning_rate", 1e-5)),
            per_device_train_batch_size=int(config_model.get("train_batch_size", 32)),
            per_device_eval_batch_size=int(config_model.get("eval_batch_size", 32)),
            num_train_epochs=int(config_model.get("num_train_epochs", 1)),
            weight_decay=float(config_model.get("weight_decay", 0.01)),
            evaluation_strategy="steps",
            eval_steps=int(config_model.get("eval_steps", 10)),
            logging_steps=int(config_model.get("logging_steps", 20)),
            save_strategy="steps",
            save_steps=int(config_model.get("save_steps", 20)),
            save_total_limit=int(config_model.get("save_total_limit", 3)),
            logging_strategy="steps",
            load_best_model_at_end=True,
            push_to_hub=False,
            group_by_length=True,
        )
        self.trainer = SigmoidTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )
        logger.info("Trainer built. Running training...")
        self.trainer.train()
        train_loss = [(i["loss"], i["step"]) for i in self.trainer.state.log_history if "loss" in i]
        eval_loss = [(i["eval_loss"], i["step"]) for i in self.trainer.state.log_history if "eval_loss" in i]
        return train_loss, eval_loss

    def predict(self) -> List[Dict[str, Union[str, float]]]:
        """
        Populates self.predictions with Dict objects.
        Each object represents the prediction results per input with the following data:
            prediction: compliant or non-compliant
            score: classifier score (0 if Stoplist fails)
            non-compliant-text: first sentence in input to fail compliance check
            (empty if compliant)
        """
        self.model = self.model.eval()
        input_strs = self.test_dataset[self.config["input_data"]["content_column"]].tolist()

        if "fairhousing" in self.config and "threshold" in self.config["fairhousing"]:
            threshold = self.config["fairhousing"]["threshold"]
            logger.info(f"Using config threshold of {threshold}.")
        else:
            threshold = 0.5
            logger.info("Threshold not found in config, using 0.5 as threshold.")

        # Build batch of sentences to run against classifier
        # Each individual sentence is checked against model
        sentences = []
        input_mapping = {}
        for input_str in input_strs:
            for sent in self.phrase_checker.get_sentences(input_str):
                sent = sent[:-1] if (sent.endswith(".") or sent.endswith("?")) else sent
                sent = sent.lower()
                sentences.append(sent)
                input_mapping[sent] = input_str

        feats = self.tokenizer(
            sentences,
            return_tensors="pt",
            max_length=self.config["input_data"]["max_length"],
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            outs = self.model(**feats)

        preds = torch.nn.functional.sigmoid(outs.logits.squeeze(1))
        # Set value of 0 or 1 to preds, based on threshold
        preds_modified = torch.where(
            preds > threshold,
            torch.tensor(1, device=self.device),
            torch.tensor(0, device=self.device),
        )

        preds = preds.cpu().numpy()
        preds_modified = preds_modified.cpu().numpy()

        # Map sentence predictions back to their original input text
        total_score = {}
        prediction = {}
        non_compliant_text = {}
        for pred, pred_modified, sent in zip(preds, preds_modified, sentences):
            # Set defaults if this is the first sentence of input being checked
            prediction.setdefault(input_mapping[sent], "compliant")
            total_score.setdefault(input_mapping[sent], [])
            non_compliant_text.setdefault(input_mapping[sent], "")
            if prediction.get(input_mapping[sent], "compliant") == "non-compliant":
                # Input already determined to be non-compliant, skip
                continue
            if self.phrase_checker.check_phrase_is_stoplist_compliant(sent):
                # Stoplist passed
                score = pred.item()
                sent_prediction = ID_TO_LABEL[pred_modified.item()]
                if sent_prediction == "non-compliant":
                    prediction[input_mapping[sent]] = "non-compliant"
                    total_score[input_mapping[sent]] = [score]
                    non_compliant_text[input_mapping[sent]] = sent
                total_score[input_mapping[sent]].append(score)
            else:
                # Stoplist failure
                logger.info(f"Stoplist check failed for sentence: {sent}")
                prediction[input_mapping[sent]] = "non-compliant"
                total_score[input_mapping[sent]] = [0]
                non_compliant_text[input_mapping[sent]] = sent

        # Build final prediction for each input
        for input_str in input_strs:
            input_prediction = {
                "prediction": prediction[input_str],
                "score": sum(total_score[input_str]) / len(total_score[input_str]),
                "non-compliant-text": non_compliant_text[input_str],
            }
            self.predictions.append(input_prediction)

        return self.predictions


class SigmoidTrainer(Trainer):
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        outs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        labels = inputs["labels"]
        loss = self.loss_fn(outs.logits.squeeze(0).reshape(labels.shape, 1), labels)
        if return_outputs:
            return loss, outs
        return loss
