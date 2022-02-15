import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, ElectraTokenizer, XLNetTokenizer, RobertaTokenizer

class CustomDataset(Dataset):

    def __init__(self, data_file, label_map, tokenizer, max_len=256, input_col='text', labels_col='labels'):
        self.label_map = label_map
        if 'bert' in tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', )
        elif 'electra' in tokenizer:
            self.tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
        elif 'xlnet' in tokenizer:
            self.tokenizer = XLNetTokenizer.from_pretrained('hfl/chinese-xlnet-base')
        elif 'roberta' in tokenizer:
            self.tokenizer = RobertaTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.data = self.get_dataFrame(data_file)#dataframe
        self.input_col = self.data[input_col]
        self.targets = self.data[labels_col]
        self.max_len = max_len

    def __len__(self):
        return len(self.input_col)

    def __getitem__(self, index):
        input_col = str(self.input_col[index])
        input_col = " ".join(input_col.split())

        # sep_token 
        # label_list = sorted(self.label_map.keys(), key=lambda v:self.label_map[v])
        # label_str = ' '.join(label_list)

        inputs = self.tokenizer.encode_plus(
            text=input_col,
            text_pair=None, 
            # text_pair=label_str,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation='only_first',
        )
        ids = inputs['input_ids']
        # print(ids)
        # print(len(ids))
        # input("=======")
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        # print('ids', torch.tensor(ids, dtype=torch.long))
        # print('mask', torch.tensor(mask, dtype=torch.long))
        # print('token_type_ids', torch.tensor(token_type_ids, dtype=torch.long))
        # print('targets', torch.tensor(self.targets[index], dtype=torch.float16))

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float16)
        }

    def labels2index(self, labels):
        """
        :params: labels 是一串字符串，包含数个标签，标签由空格分开
        """
        index = [0] * len(self.label_map)
        for label in labels.split():
            index[self.label_map[label]] = 1
        return index

    def get_dataFrame(self, data_file):
        data = pd.read_csv(data_file, encoding="UTF-8", sep='\t')
        # data['WORD_COUNT']=data['content'].apply(lambda x:len(x.split()))
        # data.hist('WORD_COUNT', bins=30)
        text_l = []
        labels_l = []
        for index, row in data.iterrows():
            text_l.append(row["content"])
            labels_l.append(self.labels2index(row['label']))
        return pd.DataFrame(data={'text': text_l, 'labels': labels_l})

def get_dataloader(data_file, label_map, tokenizer, max_len, params):
    dataset = CustomDataset(data_file, label_map, tokenizer, max_len)
    print(data_file, len(dataset))
    return DataLoader(dataset, **params)
