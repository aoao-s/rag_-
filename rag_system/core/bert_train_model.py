import re
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from base.config import get_settings,BASE_DIR
# 标签只能有一处出处（CLAUDE.md §6.4 / base/labels.py 的注释）。
# 本脚本原先自己写了一份 6 类 label2id，与 `INTENT_LABELS` 的 7 类业务意图不一致——
# 那样训出来的模型下标与推理侧、数据集、日志全部对不上，而且不会报错，只会静默错分。
from base.labels import ID2INTENT, INTENT2ID, INTENT_LABELS
# from datasets import Dataset
# from modelscope.msdatasets.dataset_cls.custom_datasets.audio.kws_nearfield_processor import padding
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torchinfo import summary

# from torch.utils.data import Dataset


def handle_command(text):
    # 处理用户评论数据
    # fit:从所有评论中学习词汇表计算 IDF
    # transform:把每条评论转换成 TF-IDF 数字向量
    # 把机器学习模型不能直接理解的中文评论，转换成模型能够处理的数字特征
    return re.sub(
        r'[^\u4e00-\u9fa5a-zA-Z0-9,.!:;\s]',
        '',
        text
    )
def load_corpus(*,frac=1,split='train'):
    # `split` 决定读哪一份：train 用来训练+验证（再切 8.5/1.5），test 是**留出集**，
    # 只在训练结束后跑一次。原实现把 train 再切三份当测试集，等于拿调过参的那批数据自评，
    # 报出来的准确率必然偏高；数据集本身就把 2 100 / 500 分好了，没有理由不用。
    train_pth=BASE_DIR/f"datasets/06_intent/intent_{split}.jsonl"
    print(train_pth)
    df = pd.read_json(train_pth,lines=True)
    # print(df)
    df=df.sample(frac=frac)
    df['query']=df['query'].map(handle_command)
    print(df['query'].map(len).agg(['max', 'min', 'mean', 'std']))
    unknown = sorted(set(df['label'].values.tolist()) - set(INTENT_LABELS))
    if unknown:
        # 认不出的标签会让 `INTENT2ID[label]` 抛 KeyError，或者在别处被静默吞掉。
        # 在装载期就报出来，比训到一半才好定位便宜得多。
        raise SystemExit(f"{train_pth.name} 里有词表外的标签：{unknown}")
    return [(text,label) for text,label in zip(df['query'].values.tolist(),df['label'].values.tolist())]
# 之前的部分Dataset都是转好的tensor,这里是字符串,所以需要我们自行定义函数来完成
# 转tensor
class WaimaiDataset(Dataset):
    # 如果想要DataLoader自动取数据,至少要实现
    # 1.__init__ 2.__len__3.__getitem__
    def __init__(self,corpus,tokenizer,max_len):
        super().__init__()
        querys = [text for text, _ in corpus]
        labels = [label for _, label in corpus]
        # 中文标签转换成整数：下标由 `INTENT_LABELS` 的顺序决定，本文件不再自己写一份
        labels = [INTENT2ID[label] for label in labels]
        self.encodings = tokenizer(
            querys,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):

        # 因为pt为要转pyTorch Tensor,会自动增加一个batch的维度,所以[0]
        # dataset返回为[94],不能是[1,94],dataloder会堆叠成一批,就出错了

        return {'input_ids':self.encodings['input_ids'][idx],
                #
                'attention_mask':self.encodings['attention_mask'][idx],
                'labels':self.labels[idx]}
# 与 §7.7 的置信度闸门同值：低于它就不许在 7 类里硬选一个，判 unknown 走 generic_clarify。
# 规则分类器、大模型路径、BERT 路径三处必须是同一个数，否则"低置信不猜"这条规则能被绕过。
MIN_CONFIDENCE: float = 0.6


def evaluate(model,data_loader,device,*,min_confidence=MIN_CONFIDENCE):
    """返回 acc / macro-F1 / unknown 比例。

    **F1 才是验收口径**（§7.7 要求 ≥0.90）：7 类样本均衡时准确率看着漂亮，
    但某一类整体失守（比如"超出范围"被并进"闲聊"）准确率几乎不动，F1 会立刻掉下来。
    原实现只报有效集准确率，看不出这件事。
    """
    model.eval()
    correct=0
    total=0
    unknown=0
    n=len(INTENT_LABELS)
    tp={i:0 for i in range(n)}
    fp={i:0 for i in range(n)}
    fn={i:0 for i in range(n)}
    with torch.no_grad():
        for bacth in data_loader:
            input_ids=bacth['input_ids'].to(device)
            attention_mask=bacth['attention_mask'].to(device)
            labels=bacth['labels'].to(device)
            outputs=model(input_ids=input_ids,
                          attention_mask=attention_mask,)
            probs=torch.softmax(outputs.logits,dim=1)
            top_prob,preddictions=torch.max(probs,dim=1)
            # 低置信 → unknown：这条样本既不算答对，也不进任何一类的预测计数
            # （它确实没给出类别，记进某类等于虚报该类精度）
            low=top_prob<min_confidence
            unknown+=int(low.sum().item())
            preddictions=preddictions.masked_fill(low,-1)
            correct+=torch.sum(preddictions==labels).item()
            total+=labels.size(0)
            for gold,pred in zip(labels.tolist(),preddictions.tolist()):
                if gold==pred:
                    tp[gold]+=1
                else:
                    fn[gold]+=1
                    if pred in fp:
                        fp[pred]+=1
    f1s=[]
    for i in range(n):
        p=tp[i]/(tp[i]+fp[i]) if tp[i]+fp[i] else 0.0
        r=tp[i]/(tp[i]+fn[i]) if tp[i]+fn[i] else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return {
        'acc':correct/total if total else 0.0,
        'macro_f1':sum(f1s)/n,
        'unknown_rate':unknown/total if total else 0.0,
    }


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    corpus=load_corpus(frac=1,split='train')
    # 2 100 条只切"训练 / 验证"两段；测试集是 `intent_test.jsonl` 那 500 条留出集，
    # 训完跑**一次**（设计文档 §3.7）。原实现从 train 里再切 10% 当测试集，
    # 报出来的数字是"拿调参用过的数据自评"，不能作为上线依据。
    train_data, valid_data = train_test_split(
        corpus,
        test_size=0.15,
        random_state=42,
        stratify=[label for _, label in corpus]
    )
    test_data = load_corpus(frac=1,split='test')
    # 加载BERT对应的词典
    tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    train_dataset=WaimaiDataset(train_data,tokenizer,24)
    # num_labels=2,告诉模型接下来需要完成2分类
    valid_dataset = WaimaiDataset(
        valid_data,
        tokenizer,
        max_len=24
    )
    test_dataset = WaimaiDataset(
        test_data,
        tokenizer,
        max_len=24
    )
    train_loder=DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=0)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    # model=(AutoModelForSequenceClassification.
    #        from_pretrained('google-bert/bert-base-chinese',num_labels=10))

    model = AutoModelForSequenceClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        # 类别数与两份映射都取自 `base/labels.py`：推理侧按 `ID2INTENT` 还原中文标签，
        # 这里若自己写一份，训出来的下标迟早与推理侧错位（而且不会报错）。
        num_labels=len(INTENT_LABELS),
        label2id=dict(INTENT2ID),
        id2label=dict(ID2INTENT),
    )
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.layer[:6].parameters():
        param.requires_grad = False
    model.to(device)
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(filter(
        lambda p: p.requires_grad,
        model.parameters()),
        lr=1e-5,weight_decay=0.01)
    # summary(
    #     model,
    #     input_data={
    #         'input_ids': torch.ones(16, 24, dtype=torch.long).to(device),
    #         'attention_mask': torch.ones(16, 24, dtype=torch.long).to(device)
    #     }
    # )
    best_f1=0.0
    save_dir = get_settings().MODELS_DIR / 'bert-intent-ft'
    for epoch in range(8):
        model.train()
        eopch_start=time.time()
        total_loss=0.0
        for bacth in train_loder:
            optimizer.zero_grad()
            # inpus['input_ids']为32句话的token编号
            input_ids=bacth['input_ids'].to(device)
            # 结构为[1,1,1,0,0]告诉模型,那些可以参与计算
            attention_mask=bacth['attention_mask'].to(device)
            label=bacth['labels'].to(device)
            #这部分返回的是SequenceClassifierOutput对象,需要.logits取出分类结果
            logits=model(input_ids=input_ids,
                         attention_mask=attention_mask)
            logits=logits.logits
            loss=loss_func(logits,label)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        valid_metrics = evaluate(
                model,
                valid_loader,
                device
            )
        epoch_end=time.time()-eopch_start
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loder):.4f}"
              f":valid_acc={valid_metrics['acc']:.4f}"
              f":valid_macro_f1={valid_metrics['macro_f1']:.4f}"
              f":valid_unknown={valid_metrics['unknown_rate']:.4f}"
              f":time={epoch_end}")
        # 按 **macro-F1** 选模型，不按准确率：验收门槛就是 F1 ≥0.90（§7.7），
        # 两类指标在均衡数据上常常同时上升，但失守的那一类只有 F1 看得出来。
        if valid_metrics['macro_f1']>best_f1:
           best_f1=valid_metrics['macro_f1']
           print(f'Best valid macro_f1={best_f1:.4f}')

           model.save_pretrained(save_dir)
    # tokenizer词表加规则
           tokenizer.save_pretrained(save_dir)

    # ---- 留出集只跑一次 --------------------------------------------------
    # 训练、选模型、调超参全程没碰过 `intent_test.jsonl`，这里的数才是能写进报告的。
    if best_f1>0.0:
        best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
        test_metrics = evaluate(best_model, test_loader, device)
        print(f"== 留出集（{len(test_data)} 条）: acc={test_metrics['acc']:.4f}"
              f" macro_f1={test_metrics['macro_f1']:.4f}"
              f" unknown={test_metrics['unknown_rate']:.4f}")
        print(f"== 门槛（macro-F1 ≥0.90 且 unknown ≤0.10）："
              f"{'通过' if test_metrics['macro_f1']>=0.90 and test_metrics['unknown_rate']<=0.10 else '未通过'}")

if __name__=='__main__':
    main()