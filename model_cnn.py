import torch
import shutil
from torch import nn
from transformers import BertModel

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class MultilabelClassifier(torch.nn.Module): 
    def __init__(self, n_classes, embedding_dim=768, filter_num=50, kernel_list=(3, 4, 5, 10, 50, 100), 
        max_seq=400, dropout=0.5, topk=5, **kargs):
        super(MultilabelClassifier, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-chinese')#('hfl/chinese-bert-wwm-ext')#('bert-base-uncased')
        # self.l2 = torch.nn.Dropout(0.1) # 0.3
        # self.l3 = torch.nn.Linear(768, n_classes)

        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        if topk==1:
            self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)), # [batch_size, filter_num, max_seq-kernel+1, 1]
                              nn.LeakyReLU(),
                              nn.MaxPool2d((max_seq - kernel + 1, 1))
                              )
                for kernel in kernel_list
            ])

        else:
            self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)), # [batch_size, filter_num, max_seq-kernel+1, 1]
                              nn.LeakyReLU()
                              )
                for kernel in kernel_list
            ])
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(filter_num * len(kernel_list), n_classes)
        ### maxk pooling
        self.topk = topk
        self.fc = nn.Linear(filter_num * len(kernel_list)*self.topk, n_classes)
        
    
    def forward(self, ids, mask, token_type_ids):
        # _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # output_2 = self.l2(output_1)
        # output = self.l3(output_2)
        # return output

        last_hidden_state, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)

        # last_hidden_state: [8, 256, 768] (batch, seq_len, embedding_dim)
        x = last_hidden_state.unsqueeze(1)     # [8, 1, 256, 768] 即(batch, channel_num, seq_len, embedding_dim)

        out = [conv(x) for conv in self.convs]
        # for a in out:
        #     print('shape of a', a.shape)
        # add maxk pooling
        if self.topk > 1:
            out = [kmax_pooling(x,2,self.topk) for x in out]

        out = torch.cat(out, dim=1)   # [batch, filter_num*len(kernel_list), 1, 1]，各通道的数据拼接在一起 # for kmax_pooling:[batch_size,filter_num*len(kernel_list), k, 1]
        # print('shape of out', out.shape)
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)        # 构建dropout层

        logits = self.fc(out)          # 结果输出[batch, n_classes]
        return logits

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    """
    resume training
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # initialize scheduler from checkpoint to scheduler
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min#.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)