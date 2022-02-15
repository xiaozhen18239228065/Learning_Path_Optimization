'''
pip install transformers==3
refer:
https://blog.csdn.net/deephub/article/details/115390956

''' 
import os
import numpy as np
import pandas as pd
from sklearn import metrics
# from sklearn.metrics import multilabel_confusion_matrix as mcm, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
# import transformers
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import data_utils
# from model import *
# from model_label_attention import *
# from model_label_attention_nocontentatten import *
from model_cnn import *
# from model_attention import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training model")
parser.add_argument("--epochs", type=int, default=30, help="total training epochs")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="initial learning rate") #
parser.add_argument("--max_len", type=int, default=400, help="max length of sequence for tokenizer")
parser.add_argument("--filter_num", type=int, default=10, help="")
parser.add_argument("--topk", type=int, default=5, help="topk maxpooling for cnn model, default maxpooling")
parser.add_argument("--label_emb_file", type=str, default='label_mean_embeddings.npy', help='the label embedding file')
parser.add_argument("--model_name", type=str, default='bert', help='base model')  #roberta#electra#xlnet#('hfl/chinese-bert-wwm-ext')#('bert-base-uncased', do_lower_case=True)
parser.add_argument("--mode", type=str, default='mean', help='pool or mean')
parser.add_argument("--cks_dir", type=str, default='models/train_model_cnn_50_0.5_maxlen400_top5_filter10_kerne3451050100-cks_temp', help='directory to store the checkpoints')
parser.add_argument("--resume", action='store_true', help='resume training from last checkpoint')
parser.set_defaults(resume=False)
parser.add_argument("--resume_checkpoint_path", type=str, default='models/train_model_cnn_50_0.5_maxlen400_top5_filter10_kerne3451050100-cks/epoch013.pt', help='resume checkpoint path')

# Defining some key variables that will be used later on in the training
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train_model(start_epochs, n_epochs, valid_loss_min_input, 
                training_loader, validation_loader, model, 
                optimizer, scheduler):
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 
    for epoch in range(start_epochs, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        print('############# Epoch {} #############'.format(epoch))
        print('training...')
        for batch_idx, data in enumerate(training_loader):
            #print('yyy epoch', batch_idx)
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            
            if batch_idx % 100 == 0:
                print("Epoch: {} Batch: {} Loss: {}".format(epoch, batch_idx, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            #print('before loss data in training', loss.item(), train_loss)
            # train_loss = int(train_loss + ((1 / (batch_idx + 1)) * (loss- train_loss)))
            #print('after loss data in training', loss.item(), train_loss)
            train_loss += float(loss.item())

        print('validation...')
        ######################    
        # validate the model #
        ######################

        model.eval()
        val_targets=[]
        val_outputs=[]
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                # valid_loss = int(valid_loss + ((1 / (batch_idx + 1)) * (loss - valid_loss)))
                valid_loss += float(loss.item())
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            eval_metrics(val_targets, val_outputs)

            # calculate average losses
            #print('before cal avg train loss', train_loss)
            train_loss = train_loss/len(training_loader)
            valid_loss = valid_loss/len(validation_loader)
            # print training/validation statistics 
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))

            is_best = False
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                # save checkpoint as best model
                is_best = True
                valid_loss_min = valid_loss


            # eval on test data
            print('-------- Eval metrics on test dataset ---------')
            test_targets, test_outputs = do_prediction(model, test_loader)
            eval_metrics(test_targets, test_outputs)

        # save checkpoint
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': valid_loss_min,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict()
        }
        # save_ckp(checkpoint, is_best, os.path.join(args.cks_dir, 'current_checkpoint.pt'), best_model_path)
        torch.save(checkpoint, os.path.join(args.cks_dir, 'epoch%03d.pt'%epoch))

        print('############# Epoch {}  Done   #############\n'.format(epoch))
        sys.stdout.flush()

    return model


def do_prediction(model, loader):
    model.eval()
   
    fin_outputs=[]
    fin_targets=[]
    with torch.no_grad():
        for _, data in enumerate(loader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_targets, fin_outputs

def eval_metrics(targets, outputs):
    targets=(np.array(targets)>0.5).astype(int)
    preds=(np.array(outputs)>0.5).astype(int)

    accuracy = metrics.accuracy_score(targets, preds)
    f1_score_micro = metrics.f1_score(targets, preds, average='micro')
    f1_score_macro = metrics.f1_score(targets, preds, average='macro')
    precision = metrics.precision_score(targets, preds, average='micro')
    recall = metrics.recall_score(targets, preds, average='micro')
    print(f"Accuracy Score = {accuracy}\n"
            f"F1 Score (Micro) = {f1_score_micro}\n" #计算所有类的recall 和 precision，由公式f1=2*(r*p)/(r+p)
            f"F1 Score (Macro) = {f1_score_macro}\n" #所有类的f1值的和除以类别数，eg:(f1+f2+f3)/3
            f"Precision (micro) = {precision}\n"
            f"Recall (micro) = {recall}")

    return metrics.precision_recall_fscore_support(targets, preds)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    axes.set_xlabel('True label')
    axes.set_ylabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)

if __name__=='__main__':
    # Sections of config

    args = parser.parse_args()

    ### dataset
    dataset_dir = '../data'

    label_file = os.path.join(dataset_dir, "label.txt")
    label_list = [w.strip() for w in open(label_file).readlines()]
    label_map = {word:idx  for idx, word in enumerate(label_list)}
    label_emb = np.load(os.path.join(dataset_dir, args.label_emb_file)) # (97, 768)
    # print(label_emb.shape)
    label_emb = torch.from_numpy(label_emb).float()

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = data_utils.get_dataloader(os.path.join(dataset_dir, "train.tsv"), 
                                    label_map, args.model_name, args.max_len, train_params)
    validation_loader = data_utils.get_dataloader(os.path.join(dataset_dir, "dev.tsv"), 
                                    label_map, args.model_name, args.max_len, test_params)
    test_loader = data_utils.get_dataloader(os.path.join(dataset_dir, "test.tsv"), 
                                    label_map, args.model_name, args.max_len, test_params)

    ### model
    model = MultilabelClassifier(n_classes=len(label_list), label_emb=label_emb, mode=args.mode, filter_num=args.filter_num,
                                    pretrain=args.model_name, max_seq=args.max_len, topk=args.topk)

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    # optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    total_steps = len(training_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    ### training
    # cks_dir = 'mean-cks'
    if not os.path.exists(args.cks_dir):
        os.makedirs(args.cks_dir)
    # checkpoint_path = os.path.join(args.cks_dir, 'current_checkpoint.pt')
    # best_model_path = os.path.join(args.cks_dir, 'best_model.pt')
    start_epoch = 1
    valid_loss_min = np.Inf
    if args.resume and os.path.exists(args.resume_checkpoint_path):
        print('load model from', args.resume_checkpoint_path)
        model, optimizer, epoch, valid_loss_min = load_ckp(args.resume_checkpoint_path, model, optimizer, scheduler) #scheduler, 
        start_epoch = epoch + 1
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    trained_model = train_model(start_epoch, args.epochs, valid_loss_min, training_loader, validation_loader, model, 
                                    optimizer, scheduler)

    ### evaluation
    # print('-------- Eval metrics on test dataset (best model) ---------')
    # checkpoint = torch.load(best_model_path)
    # # initialize state_dict from checkpoint to model
    # model.load_state_dict(checkpoint['state_dict'])
    # test_targets, test_outputs = do_prediction(model, test_loader)
    # eval_metrics(test_targets, test_outputs)

    # cm = mcm(val_targets, val_preds)#混淆矩阵
    # print(classification_report(val_targets, val_preds,target_names=label_list))

    # sns.set_style("whitegrid")
    # fig, ax = plt.subplots(4, 2, figsize=(12, 7))
    # for axes, cfs_matrix, label in zip(ax.flatten(), cm, cm_labels):
    #   print_confusion_matrix(cfs_matrix, axes, label, ["1", "0"])
        
    # fig.tight_layout()
    # plt.show()

'''
4 epoch
Accuracy Score = 0.6144558108334731
F1 Score (Micro) = 0.9001071507995876
F1 Score (Macro) = 0.8150760649041422
Precision (micro) = 0.9187750216682488
Recall (micro) = 0.8821827692795434
'''

