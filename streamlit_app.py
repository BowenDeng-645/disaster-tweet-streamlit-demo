# -*- coding: utf-8 -*-
"""
streamlit_app.py —— 用于 Streamlit Community Cloud (share.streamlit.io) 的部署入口

背景：Hugging Face Spaces 的 Gradio/Docker SDK（能跑Python后端的Space）目前
需要付费的 Team/Enterprise 计划才能使用，免费账号无法创建。因此改用
Streamlit Community Cloud ——它对公开仓库完全免费、长期有效，不需要任何付费。

模型权重不随这个小仓库一起提交（避免GitHub仓库过大），而是从免费的
Hugging Face 模型仓库（注意：是Model仓库，不是Space，不受付费限制）
https://huggingface.co/binfei99/disaster-tweet-distilbert
在应用启动时自动下载加载。
"""

import streamlit as st
import torch
from transformers import AutoTokenizer

from modeling_bert import BertForTextClassification

BASE_MODEL_NAME = "distilbert-base-uncased"
# 免费的 Hugging Face 模型仓库（Model repo，不是Space，始终免费）
HF_MODEL_REPO = "binfei99/disaster-tweet-distilbert"
MAX_SEQ_LENGTH = 128
LABEL_NAMES = {0: "Not Disaster (非灾难推文)", 1: "Disaster (真实灾难推文)"}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
    model = BertForTextClassification.load(
        HF_MODEL_REPO, model_name_or_path=BASE_MODEL_NAME, num_labels=len(LABEL_NAMES)
    )
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs["logits"], dim=-1)[0]
    return {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}


st.set_page_config(page_title="推文真假灾难分类 Demo (BERT微调)")
st.title("推文真假灾难分类 Demo (BERT微调)")
st.write(
    "基于 Kaggle 'Real or Not? NLP with Disaster Tweets' 数据集微调的文本分类模型。"
    "输入一条推文文本，模型会判断它是否描述了真实的灾难事件。"
)

tokenizer, model = load_model()

examples = [
    "Forest fire near La Ronge Sask. Canada",
    "I love the smell of fresh coffee in the morning",
    "Just happened a terrible car crash on the highway",
]
choice = st.selectbox("快速试试示例：", ["(自己输入)"] + examples)
default_text = "" if choice == "(自己输入)" else choice

text = st.text_area("输入文本 (Tweet)", value=default_text, height=100)

if st.button("Submit") and text.strip():
    result = predict(text, tokenizer, model)
    st.subheader("分类结果")
    for label, prob in sorted(result.items(), key=lambda x: -x[1]):
        st.write(f"{label}: {prob:.1%}")
        st.progress(prob)
