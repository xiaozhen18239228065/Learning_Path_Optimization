import os 
import numpy as np
import torch

# import tensorflow as tf
# from transformer.model import TransformerClassifier
# from utils.preprocess import load_testcnn_data
# from transformer import train as transformer_utils

from main import do_prediction, eval_metrics
from model_cnn import *
# from model import *
import data_utils
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    ### bert dataset
    dataset_dir = '../data'

    label_file = os.path.join(dataset_dir, "label.txt")
    label_list = [w.strip() for w in open(label_file).readlines()]
    label_map = {word:idx  for idx, word in enumerate(label_list)}

    test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }

    validation_loader = data_utils.get_dataloader(os.path.join(dataset_dir, "dev.tsv"), 
                                    label_map, 'bert-base-chinese', 400, test_params)
    test_loader = data_utils.get_dataloader(os.path.join(dataset_dir, "test.tsv"), 
                                    label_map, 'bert-base-chinese', 400, test_params)

    ### bert model
    bert_model = MultilabelClassifier(n_classes=len(label_list), mode='mean', max_seq=400, topk=5)
    if torch.cuda.device_count() > 1:
        bert_model = torch.nn.DataParallel(bert_model)
    bert_model.to(torch.device("cuda"))

    # load check point
    bert_checkpoint_fpath = 'models/train_model_cnn_50_0.5_maxlen400_top5_kerne3451050100-cks/epoch025.pt' #'models/model_mean_adamw_maxlen400/epoch016.pt'#
    checkpoint = torch.load(bert_checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    bert_model.load_state_dict(checkpoint['state_dict'])

    # bert_model.eval()

    bert_dev_targets, bert_dev_outputs = do_prediction(bert_model, validation_loader)
    bert_dev_precision, bert_dev_recall, bert_dev_f1, bert_dev_support = eval_metrics(bert_dev_targets, bert_dev_outputs)

    bert_test_targets, bert_test_outputs = do_prediction(bert_model, test_loader)
    bert_test_precision, bert_test_recall, bert_test_f1, bert_test_support = eval_metrics(bert_test_targets, bert_test_outputs)

    # print('bert_dev_precision = ', list(bert_dev_precision))
    # print('bert_dev_recall = ', list(bert_dev_recall))
    # print('bert_dev_f1 = ', list(bert_dev_f1))
    print('bert_dev_support = ', list(bert_dev_support))

    # print('bert_test_precision = ', list(bert_test_precision))
    # print('bert_test_recall = ', list(bert_test_recall))
    # print('bert_test_f1 = ', list(bert_test_f1))
    print('bert_test_support = ', list(bert_test_support))



    # ### transformer dataset
    # train_x, dev_x, test_x, train_y, dev_y, test_y = load_testcnn_data()
    # train_dataset = transformer_utils.load_dataset(train_x, train_y)

    # # 模型
    # transformer_model = TransformerClassifier(4, 128, 8, 512, 50000, 10000, 97, 0.1)


