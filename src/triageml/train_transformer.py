"""Fine-tune a DistilBERT classifier with PyTorch + Hugging Face Transformers.

Heavy dependencies (torch, transformers) are optional and imported lazily, so
the baseline pipeline, the API, and CI all run without installing them. Install
the extras when you want to train this model:

    pip install -e ".[transformer]"
    python -m triageml.train_transformer

On CPU this is slow; use a GPU machine (or a cloud GPU instance) for real runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import settings
from .data import load_split


def _require_torch_stack():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        from datasets import Dataset as HFDataset  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "Transformer training requires the optional extras. Install with:\n"
            '    pip install -e ".[transformer]"'
        ) from exc


def train(output_dir: Path | None = None) -> dict:
    _require_torch_stack()

    import numpy as np
    from datasets import Dataset as HFDataset
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    output_dir = output_dir or settings.transformer_path
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_split()
    encoder = LabelEncoder().fit(data.y_train)
    labels = list(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(settings.transformer_base_model)

    def to_hf(texts: list[str], ys: list[str]) -> HFDataset:
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=settings.max_length,
        )
        enc["labels"] = encoder.transform(ys).tolist()
        return HFDataset.from_dict(enc)

    train_ds = to_hf(data.X_train, data.y_train)
    eval_ds = to_hf(data.X_test, data.y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.transformer_base_model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={lab: i for i, lab in enumerate(labels)},
    )

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(gold, preds, average="macro")}

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "model": "distilbert_finetuned",
        "macro_f1": round(float(eval_metrics["eval_macro_f1"]), 4),
        "n_train": data.n_train,
        "n_test": data.n_test,
        "labels": labels,
        "artifact": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
