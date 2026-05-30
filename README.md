# RoBERTa-Hotel-Sentiment-Analysis

## 项目简介
基于中文RoBERTa预训练模型实现酒店用户评论二分类情感分析，完成数据集预处理、模型微调、FastAPI工程化接口部署全链路NLP项目。

## 技术栈
Python、PyTorch、HuggingFace Transformers、RoBERTa-wwm-ext、Trainer、FastAPI、Uvicorn

## 项目结构
```
RoBERTa-Hotel-Sentiment-Analysis/
├── train.py              # 数据集处理+模型微调训练
├── app.py                # FastAPI在线推理接口
├── test.py               # 离线单文本测试
├── requirements.txt      # 环境依赖清单
├── .gitignore            # 忽略缓存、大模型权重文件
└── README.md             # 项目说明文档
```
> 注：训练生成的best_sentiment_model权重文件夹因体积过大，未上传至仓库，本地运行train.py自动生成

## 模型指标
测试集准确率Acc：91% | 加权F1：0.91，正负评论识别精准。

## 快速运行
1. 安装项目依赖
```bash
pip install -r requirements.txt
```

2. 启动模型训练
```bash
python train.py
```

3. 启动API服务
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
> 服务启动成功后，浏览器访问调试文档：`http://127.0.0.1:8000/docs`
