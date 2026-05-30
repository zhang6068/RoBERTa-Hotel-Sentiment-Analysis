import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")

# ===================== 加载训练好的模型 =====================
MODEL_PATH = "./best_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 自动使用 GPU / CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 标签映射（从模型里自动读取，你训练时写的 id2label）
id2label = model.config.id2label


# ===================== 预测函数 =====================
def predict(text):
    model.eval()
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        pred_label = id2label[pred_id]  # 自动转成文字标签
    return pred_id, pred_label


# ===================== 测试 =====================
if __name__ == "__main__":
    print("===== 中文情感分类测试=====")

    texts = [
        "酒店服务非常差，房间又冷又潮",
        "位置还行，价格一般，凑合住",
        "这家餐厅味道太难吃了",
        "物流太慢了，包装还破了，很不满意",
        "超级不好用，质量不好，强烈不推荐！"
    ]

    for t in texts:
        pred_id, pred_label = predict(t)
        print(f"\n文本：{t}")
        print(f"预测结果：{pred_id} → {pred_label}")