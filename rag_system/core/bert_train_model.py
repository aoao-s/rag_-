import re
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from base.config import get_settings,BASE_DIR
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
def load_corpus(*,frac=1):
    train_pth=BASE_DIR/"datasets/06_intent/intent_train.jsonl"
    print(train_pth)
    df = pd.read_json(train_pth,lines=True)
    # print(df)
    df=df.sample(frac=frac)
    df['query']=df['query'].map(handle_command)
    print(df['query'].map(len).agg(['max', 'min', 'mean', 'std']))
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
        # 中文标签转换成整数
        label2id = {
            "学习方法": 0,
            "教案素材": 1,
            '备课辅助':2,
            '知识点查询':3,
            '题目讲解':4,
            '闲聊':5
        }
        labels = [label2id[label] for label in labels]
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
def evaluate(model,data_loader,device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bacth in data_loader:
            input_ids=bacth['input_ids'].to(device)
            attention_mask=bacth['attention_mask'].to(device)
            labels=bacth['labels'].to(device)
            outputs=model(input_ids=input_ids,
                          attention_mask=attention_mask,)
            preddictions=torch.argmax(outputs.logits,dim=1)
            correct+=torch.sum(preddictions==labels).item()
            total+=labels.size(0)
    return correct/total


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    corpus=load_corpus(frac=1)
    train_data, temp_data = train_test_split(
        corpus,
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in corpus]
    )

    valid_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        stratify=[label for _, label in temp_data]
    )
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
        num_labels=6,
        label2id={
            "学习方法": 0,
            "教案素材": 1,
            '备课辅助':2,
            '知识点查询':3,
            '题目讲解':4,
            '闲聊':5
        },
        id2label={
            0:"学习方法",
            1:"教案素材",
            2:'备课辅助',
            3:'知识点查询',
            4:'题目讲解',
            5:'闲聊'
        }
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
    best_acc=0.0
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
        valid_acc = evaluate(
                model,
                valid_loader,
                device
            )
        epoch_end=time.time()-eopch_start
        print(f'Epoch {epoch+1}: loss={total_loss/len(train_loder):.4f}:valid_acc={valid_acc:.4f}:time={epoch_end}')
    # 模型学习参数
        if valid_acc>best_acc:
           best_acc=valid_acc
           print(f'Best valid acc={best_acc:.4f}')

           model.save_pretrained(get_settings().MODELS_DIR / 'bert-intent-ft')
    # tokenizer词表加规则
           tokenizer.save_pretrained(get_settings().MODELS_DIR / 'bert-intent-ft')

if __name__=='__main__':
    main()