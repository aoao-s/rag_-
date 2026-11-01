import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from rag_system.core.bert_train_model import MIN_CONFIDENCE, handle_command, WaimaiDataset
from base.config import get_settings,BASE_DIR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# 标签映射只认 `base/labels.py` 这一份。本文件原先自己写了 6 类 label2id，
# 与训练脚本、推理侧、数据集三方都已经对不上——7 类模型跑在这份映射上
# 要么 KeyError，要么把"超出范围"错认成"闲聊"。
from base.labels import ID2INTENT, INTENT2ID, INTENT_LABELS
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
    unknown_labels = set(test_labels) - set(INTENT2ID)
    if unknown_labels:
        raise ValueError(f"发现未知标签：{unknown_labels}")

    # 必须是 ndarray 而不是 list：下面用 `committed`（numpy 布尔掩码）去筛，
    # 而**布尔数组不能索引 Python list**——会抛
    # `TypeError: only integer scalar arrays can be converted to a scalar index`。
    true_ids = np.array([INTENT2ID[label] for label in test_labels])
    tokenizer = AutoTokenizer.from_pretrained(get_settings().MODELS_DIR / 'bert-intent-ft')
    model=AutoModelForSequenceClassification.from_pretrained(get_settings().MODELS_DIR / 'bert-intent-ft')
    # 检查点自带的标签映射必须与 `base/labels.py` 完全一致。**这个校验不能省**：
    # 下标错位时脚本照跑、不报错，只是数字很难看——本机就踩过，磁盘上那份检查点还是
    # 6 类（学习方法/教案素材/备课辅助/知识点查询/题目讲解/闲聊）时训的，
    # 拿它测 7 类数据得到 accuracy 0.395，混淆矩阵里金标「教案制作」一条都没落在 0；
    # 没有这行提示，只能靠人去猜"是模型不行还是映射错了"。
    checkpoint_labels = {int(k): v for k, v in model.config.id2label.items()}
    if checkpoint_labels != dict(ID2INTENT):
        raise SystemExit(
            f"检查点 {get_settings().MODELS_DIR / 'bert-intent-ft'} 的标签映射与 base/labels.py 不一致：\n"
            f"  检查点：{checkpoint_labels}\n"
            f"  期望：  {dict(ID2INTENT)}\n"
            "先跑 rag_system/core/bert_train_model.py 重新训练，再跑本脚本。"
        )
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
    # 低置信判 unknown（与训练脚本、与 `understanding/intent.py` 是同一个门槛）：
    # 只报"强制选一类"的 F1 会掩盖模型在该弃权时硬猜的那部分，报出来的数偏乐观。
    committed = probas >= MIN_CONFIDENCE
    print('unknown 比例:', float((~committed).sum() / len(labels)))
    print(f'accuracy:',accuracy_score(true_ids[committed], labels[committed]))
    # 两边都得是整数下标才能比：`test_labels` 是中文标签，`labels` 是类别下标，
    # 直接 zip 起来恒不相等（原实现因此永远打印 0.0，而且不报错）。
    correct_count=int((true_ids[committed] == labels[committed]).sum())
    print(f'正确的数量:',correct_count/max(int(committed.sum()),1))
    # labels 显式给全 7 类：弃权掉的类按 0 参与宏平均，否则弃权会被算成"该类不存在"
    print(f'f1',f1_score(
        true_ids[committed], labels[committed], average='macro',
        labels=list(range(len(INTENT_LABELS))), zero_division=0,
    ))
    print(f'混淆矩阵',confusion_matrix(
        true_ids[committed],
        labels[committed],
        labels=list(range(len(INTENT_LABELS))),
    ))
if __name__ == '__main__':
    main()

