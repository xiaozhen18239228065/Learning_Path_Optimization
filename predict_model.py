import os 
import numpy as np
import torch

# import tensorflow as tf
# from transformer.model import TransformerClassifier
# from utils.preprocess import load_testcnn_data
# from transformer import train as transformer_utils

from main import do_prediction, eval_metrics
# from model_cnn import *
from model import *
import data_utils
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = 'electra'

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
                                    label_map, model_name, 400, test_params)
    test_loader = data_utils.get_dataloader(os.path.join(dataset_dir, "test.tsv"), 
                                    label_map, model_name, 400, test_params)

    ### bert model
    model = MultilabelClassifier(n_classes=len(label_list), mode='mean', pretrain=model_name, max_seq=400)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(torch.device("cuda"))

    # load check point
    checkpoint_fpath = 'models/model_electra/epoch010.pt'#'models/train_model_cnn_50_0.5_maxlen400_top5_kerne3451050100-cks/epoch025.pt' #'models/model_mean_adamw_maxlen400/epoch016.pt'#
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])

    # model.eval()

    dev_targets, dev_outputs = do_prediction(model, validation_loader)
    dev_precision, dev_recall, dev_f1, dev_support = eval_metrics(dev_targets, dev_outputs)

    test_targets, test_outputs = do_prediction(model, test_loader)
    test_precision, test_recall, test_f1, test_support = eval_metrics(test_targets, test_outputs)

    print(model_name+'_dev_precision = ', list(dev_precision))
    print(model_name+'_dev_recall = ', list(dev_recall))
    print(model_name+'_dev_f1 = ', list(dev_f1))
    print(model_name+'_test_precision = ', list(test_precision))
    print(model_name+'_test_recall = ', list(test_recall))
    print(model_name+'_test_f1 = ', list(test_f1))

    print('dev_support = ', list(dev_support))
    print('test_support = ', list(test_support))



    # ### transformer dataset
    # train_x, dev_x, test_x, train_y, dev_y, test_y = load_testcnn_data()
    # train_dataset = transformer_utils.load_dataset(train_x, train_y)

    # # 模型
    # transformer_model = TransformerClassifier(4, 128, 8, 512, 50000, 10000, 97, 0.1)


