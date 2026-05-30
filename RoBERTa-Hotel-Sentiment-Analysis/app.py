from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 初始化 FastAPI 应用
app = FastAPI(title="中文情感分类模型服务", version="1.0")

# ===================== 加载模型（全局加载一次）=====================
MODEL_PATH = "./best_sentiment_model"  # 你的模型路径

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # 推理模式
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ===================== 预测接口 =====================
@app.get("/sentiment")
def predict_sentiment(text: str):
    # 1. 编码
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 2. 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. 解析结果
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    score = torch.softmax(outputs.logits, dim=-1).max().item()
    label = "POSITIVE" if pred_id == 1 else "NEGATIVE"

    # 4. 返回结果
    return {
        "text": text,
        "label_id": pred_id,
        "label": label,
        "confidence": round(float(score), 3)
    }

# ===================== 健康检查 =====================
@app.get("/")
def root():
    return {"message": "情感分类服务运行成功！", "model": "RoBERTa-wwm-ext"}