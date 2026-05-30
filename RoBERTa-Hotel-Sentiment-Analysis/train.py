import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
import torch.nn as nn
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# ===================== 1. 配置 =====================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./sentiment_model"
NUM_LABELS = 2  # 二分类 ✅


# ===================== 2. 读取 TSV 数据（已修复） =====================
def load_tsv_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头 sentence\tlabel
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue

            text, label = parts
            label = int(label)
            data.append({"text": text, "label": label})

    return pd.DataFrame(data)


train_df = load_tsv_data("data/train.tsv")
test_df = load_tsv_data("data/dev.tsv")

# 划分验证集
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 转 Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ===================== 3. 分词 =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===================== 4. 模型（二分类，标签正确） =====================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS,
    id2label={0: "Negative", 1: "POSITIVE"},
    label2id={"Negative": 0, "POSITIVE": 1},
)

# ===================== 5. 评估指标 =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ===================== 6. 训练参数 =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_only_model=True,

    fp16=True,
    dataloader_num_workers=0,
    gradient_accumulation_steps=1,
    optim="adamw_torch_fused"
)

# ===================== 7. 训练 =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# ===================== 8. 测试集评估 =====================
test_result = trainer.evaluate(test_dataset)
print("\n===== 测试集结果（二分类 0=负面,1=正面）=====")
print(f"准确率: {test_result['eval_accuracy']:.4f}")
print(f"F1分数: {test_result['eval_f1']:.4f}")

# ===================== 9. 保存模型 =====================
trainer.save_model("./best_sentiment_model")
tokenizer.save_pretrained("./best_sentiment_model")