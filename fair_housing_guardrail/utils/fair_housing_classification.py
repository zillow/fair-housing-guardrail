import logging
from typing import Dict, List, Tuple, Union

import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from fair_housing_guardrail.utils.helper import (
    compute_metrics,
    load_model,
    load_phrase_checker,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FairHousingGuardrailClassification:
    def __init__(
        self,
        config={},
        model=None,
        tokenizer=None,
        train_data=None,
        test_data=None,
        LABEL_TO_ID=None,
        ID_TO_LABEL=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.IS_BINARY = config["input_data"]["is_binary"]
        num_labels = 1 if self.IS_BINARY else len(LABEL_TO_ID)
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = load_model(self.config, num_labels)
            self.model = self.model.to(self.device)
        self.phrase_checker = load_phrase_checker(self.config["input_data"]["stop_list_path"])
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.predictions = []
        self.LABEL_TO_ID = LABEL_TO_ID
        self.ID_TO_LABEL = ID_TO_LABEL

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
            eval_strategy="steps",
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
        trainer_class = BinaryTrainer if self.IS_BINARY else CrossEntropyTrainer

        self.trainer = trainer_class(
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
        train_loss = [
            (i["loss"], i["step"]) for i in self.trainer.state.log_history if "loss" in i
        ]
        eval_loss = [
            (i["eval_loss"], i["step"]) for i in self.trainer.state.log_history if "eval_loss" in i
        ]
        return train_loss, eval_loss

    def predict(self) -> List[Dict[str, Union[str, float]]]:
        """
        Populates self.predictions with Dict objects.
        Each object represents the prediction results per input with the following data:
            prediction: predicted class label
            score: classifier confidence (probability)
            non-compliant-text: first sentence in input to fail compliance check
                (empty if compliant)
        """
        self.model = self.model.eval()
        input_strs = self.test_dataset[self.config["input_data"]["content_column"]].tolist()

        # Build batch of sentences to run against classifier
        sentences = []
        input_mapping = {}  # Maps sentence index to input string index
        for input_str_idx, input_str in enumerate(input_strs):
            for sent in self.phrase_checker.get_sentences(input_str):
                sent = sent[:-1] if (sent.endswith(".") or sent.endswith("?")) else sent
                sent = sent.lower()
                sentences.append(sent)
                input_mapping[len(sentences) - 1] = input_str_idx

        # Initialize dictionaries with default values for all input indices
        predictions = {idx: "compliant" for idx in range(len(input_strs))}
        total_scores = {idx: [] for idx in range(len(input_strs))}
        non_compliant_text = {idx: "" for idx in range(len(input_strs))}

        feats = self.tokenizer(
            sentences,
            return_tensors="pt",
            max_length=self.config["input_data"]["max_length"],
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            outs = self.model(**feats)

        # Get predictions and probabilities
        if self.IS_BINARY:
            # For binary, we need to get the threshold from the config
            if "fairhousing" in self.config and "threshold" in self.config["fairhousing"]:
                threshold = self.config["fairhousing"]["threshold"]
            else:
                logger.info("Threshold not found in config, using 0.5 as threshold")
                threshold = 0.5

            probs = torch.sigmoid(outs.logits.squeeze(-1))
            preds = torch.where(
                probs > threshold,
                1,
                0,
            )
        else:
            probs = torch.nn.functional.softmax(outs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        preds = preds.cpu().numpy()
        probs = probs.cpu().numpy()

        # Map sentence predictions back to their original input text
        for i, (sent, pred, prob) in enumerate(zip(sentences, preds, probs)):
            input_str_idx = input_mapping[i]
            # If we've already found a non-compliant sentence for this input, skip
            if "non-compliant" in predictions.get(input_str_idx, "compliant"):
                continue
            if self.phrase_checker.check_phrase_is_stoplist_compliant(sent):
                # Stoplist passed
                label = self.ID_TO_LABEL[pred]
                score = float(prob) if self.IS_BINARY else float(prob[pred])
                if "non-compliant" in label:
                    predictions[input_str_idx] = label
                    total_scores[input_str_idx] = [score]
                    non_compliant_text[input_str_idx] = sent
                    continue
                total_scores[input_str_idx].append(score)
            else:
                # Stoplist failure
                logger.info(f"Stoplist check failed for sentence: {sent}")
                predictions[input_str_idx] = (
                    "non-compliant" if self.IS_BINARY else "non-compliant-stoplist"
                )
                total_scores[input_str_idx] = [0]
                non_compliant_text[input_str_idx] = sent

        # Build final prediction for each input
        for input_str_idx in range(len(input_strs)):
            input_prediction = {
                "prediction": predictions[input_str_idx],
                "score": (
                    sum(total_scores[input_str_idx]) / len(total_scores[input_str_idx])
                    if total_scores[input_str_idx]
                    else 1.0
                ),
                "non-compliant-text": non_compliant_text[input_str_idx],
            }
            self.predictions.append(input_prediction)

        return self.predictions


class CrossEntropyTrainer(Trainer):
    loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        outs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        labels = inputs["labels"].long()
        loss = self.loss_fn(outs.logits, labels)
        if return_outputs:
            return loss, outs
        return loss


class BinaryTrainer(Trainer):
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        outs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        labels = inputs["labels"].float()
        loss = self.loss_fn(outs.logits.squeeze(1), labels)
        if return_outputs:
            return loss, outs
        return loss
