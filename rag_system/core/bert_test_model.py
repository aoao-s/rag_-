import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from rag_system.core.bert_train_model import handle_command,WaimaiDataset
from base.config import get_settings,BASE_DIR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
label2id={
        "学习方法": 0,
        "教案素材": 1,
        '备课辅助':2,
        '知识点查询':3,
        '题目讲解':4,
        '闲聊':5
        }
id2label={
        0:"学习方法",
        1:"教案素材",
        2:'备课辅助',
        3:'知识点查询',
        4:'题目讲解',
        5:'闲聊'
        }
def load_corpus(*,frac=1):
    train_pth=BASE_DIR/"datasets/06_intent/intent_test.jsonl"
    df = pd.read_json(train_pth,lines=True)
    # print(df)
    df=df.sample(frac=frac)
    df['query']=df['query'].map(handle_command)
    print(df['query'].map(len).agg(['max', 'min', 'mean', 'std']))
    return [(text,label) for text,label in zip(df['query'].values.tolist(),df['label'].values.tolist())]
def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    corpus = load_corpus(frac=1)
    tests = [text for text, label in corpus]
    test_labels = [label for text, label in corpus]
    # 中文标签转成数字
    unknown_labels = set(test_labels) - set(label2id)
    if unknown_labels:
        raise ValueError(f"发现未知标签：{unknown_labels}")

    true_ids = [label2id[label] for label in test_labels]
    tokenizer = AutoTokenizer.from_pretrained(get_settings().MODELS_DIR / 'bert-intent-ft')
    model=AutoModelForSequenceClassification.from_pretrained(get_settings().MODELS_DIR / 'bert-intent-ft')
    model.to(device)
    inputs=tokenizer(
        tests,
        return_tensors='pt',
        # 如果文本超过 max_length，就截断
        truncation=True,
        # 每条文本最终最多保留 多少 个 Token
        max_length=24,
        padding='max_length',
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    model.eval()
    # 现在只是让模型预测，不训练模型
    with torch.inference_mode():
        # 把测试文本送进 BERT
        output = model(input_ids, attention_mask)
    # output为大概为[1000,10],1000条数据的2个分类的得分(原始分数)
    probas = torch.softmax(output.logits, dim=1)
    # 得分找概率,dim=1.沿着1000的方向
    # max返回最大值及最大值的位置:probas概率,indices类型
    probas, indices = torch.max(probas, dim=1)

    probas = np.ravel(probas.cpu().numpy())
    labels = np.ravel(indices.cpu().numpy())
    print(f'accuracy:',accuracy_score(true_ids, labels))
    correct_count=sum(true_label==pred_labels for true_label,pred_labels in zip(test_labels, labels))
    print(f'正确的数量:',correct_count/len(true_ids))
    print(f'f1',f1_score(true_ids, labels,average='macro'))
    print(f'混淆矩阵',confusion_matrix(
        true_ids,
        labels
    ))
if __name__ == '__main__':
    main()

