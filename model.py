import torch
import shutil
from torch import nn
from transformers import BertModel, ElectraModel, XLNetModel, RobertaModel

class MultilabelClassifier(torch.nn.Module):
    def __init__(self, n_classes, dropout=0.1, mode='pool', pretrain='bert', **kwargs):
        super(MultilabelClassifier, self).__init__()
        if 'bert' in pretrain:
            # print('load bert')
            self.l1 = BertModel.from_pretrained('bert-base-chinese')#('hfl/chinese-bert-wwm-ext')#('bert-base-uncased')
        elif 'electra' in pretrain:
            # print('load electra')
            self.l1 = ElectraModel.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
        elif 'xlnet' in pretrain:
            self.l1 = XLNetModel.from_pretrained('hfl/chinese-xlnet-base')
        elif 'roberta' in pretrain:
            self.l1 = RobertaModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.l2 = torch.nn.Dropout(dropout) # 0.3
        self.l3 = torch.nn.Linear(768, n_classes)
        self.mode = mode
        
    
    def forward(self, ids, mask, token_type_ids):
        if self.mode == 'pool':
            _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        elif self.mode == 'mean':
            outputs = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
            if type(outputs) == tuple:
                last_hidden_state = outputs[0]
            else:
                last_hidden_state = outputs
            output_1 = torch.mean(last_hidden_state, dim=1)
        else:
            raise Exception('Invalid Mode'+self.mode)
        
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output



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
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item() #scheduler, 

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