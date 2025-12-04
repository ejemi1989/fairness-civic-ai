
import os
import yaml
import mlflow
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from preprocess import load_config, prepare_dataframes, get_tokenizer, tokenize
from data_ingest import ingest
from dataset import HFEncodedDataset  # we need to add this helper below
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# create HFEncodedDataset
# Put this into src/dataset.py (we'll create it inline here for simplicity)
# but to keep structure, assume `dataset.py` exists with HFEncodedDataset definition

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary")
    }

def run_train(config_path="config.yaml"):
    cfg = load_config(config_path)
    df = ingest(config_path)
    train_df, test_df = prepare_dataframes(df, cfg)
    tokenizer = get_tokenizer(cfg["model"]["hf_model"], cfg["model"]["max_length"])

    train_enc = tokenize(train_df, tokenizer, cfg["dataset"]["text_col"], cfg["model"]["max_length"])
    test_enc = tokenize(test_df, tokenizer, cfg["dataset"]["text_col"], cfg["model"]["max_length"])
    train_labels = train_df[cfg["dataset"]["label_col"]].astype(int).tolist()
    test_labels = test_df[cfg["dataset"]["label_col"]].astype(int).tolist()

    # Lazy import dataset class here to avoid repeated code
    from dataset import HFEncodedDataset
    train_dataset = HFEncodedDataset(train_enc, train_labels)
    eval_dataset = HFEncodedDataset(test_enc, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["hf_model"], num_labels=cfg["model"]["num_labels"])

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
        evaluation_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # MLflow logging
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({
            "model": cfg["model"]["hf_model"],
            "epochs": cfg["training"]["num_train_epochs"],
            "batch_size": cfg["training"]["per_device_train_batch_size"]
        })
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        # save model artifact
        model_path = os.path.join(cfg["training"]["output_dir"], "best_model")
        trainer.save_model(model_path)
        mlflow.pytorch.log_model(trainer.model, "hf_model")

    return model_path, train_df, test_df

if __name__ == "__main__":
    run_train()
