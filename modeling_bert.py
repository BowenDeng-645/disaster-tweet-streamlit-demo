# -*- coding: utf-8 -*-
"""
modeling_bert.py —— 模型结构模块 (解耦版本)

对应教程 dive-tuning.ipynb 2.1节："加载模型(modeling_bert.py)"

设计思路：在 BERT/DistilBERT 等预训练编码器之上，接一个简单的
分类头 (dropout + linear)，实现一个可定制、可读性强的文本分类模型，
而不是直接黑箱调用 AutoModelForSequenceClassification，方便学习
"预训练模型 + 下游任务头" 的组合方式。
"""

import inspect

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertForTextClassification(nn.Module):
    """BERT/DistilBERT 系列预训练模型 + 分类头"""

    def __init__(self, model_name_or_path: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def _pool(self, outputs, attention_mask):
        """兼容不同模型结构(BERT有pooler_output，DistilBERT没有)，
        统一取[CLS] token对应的表示作为句向量。"""
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        # 没有pooler时（如DistilBERT），取最后一层的[CLS] token (第0个位置)
        return outputs.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 不是所有模型都接受 token_type_ids（如DistilBERT不支持）
        accepted_params = inspect.signature(self.encoder.forward).parameters
        if token_type_ids is not None and "token_type_ids" in accepted_params:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        pooled = self._pool(outputs, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def save(self, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        self.config.save_pretrained(output_dir)

    @classmethod
    def load(cls, output_dir, model_name_or_path=None, num_labels=2):
        """从本地目录 或 Hugging Face Hub 的模型仓库(repo_id) 加载权重。

        output_dir 既可以是本地路径(如 'checkpoint')，也可以是Hub上的
        repo_id(如 'binfei99/disaster-tweet-distilbert')——后者常用于
        Streamlit Cloud等不方便打包大文件的部署场景。
        """
        import os

        name = model_name_or_path or output_dir
        model = cls(name, num_labels=num_labels)

        local_state_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(local_state_path):
            state_path = local_state_path
        else:
            # 不是本地路径，尝试当作 Hugging Face Hub 的模型仓库下载
            from huggingface_hub import hf_hub_download

            state_path = hf_hub_download(repo_id=output_dir, filename="pytorch_model.bin")

        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        return model
