import torch
from transformers import BertTokenizer, BertModel
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
model.eval()
mean_embeddings = []
pool_embeddings = []

label_file_path = '../data/label.txt'
with torch.no_grad():
    with open(label_file_path) as f:
        for line in f:
            label_name = line.strip()
            print(label_name)
            input_ids = torch.tensor(tokenizer.encode(label_name)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids)
            mean_embedding = torch.mean(outputs[0], dim=1)
            pool_embedding = outputs[1]
            # print('mean_embedding shape', mean_embedding.shape) # [1, 768]
            # print('pool_embedding shape', pool_embedding.shape) # [1, 768]

            mean_embeddings.append(mean_embedding)
            pool_embeddings.append(pool_embedding)
            # break

    mean_embeddings = np.array(torch.cat(mean_embeddings, 0))
    pool_embeddings = np.array(torch.cat(pool_embeddings, 0))
    print('mean_embeddings shape', mean_embeddings.shape)
    print('pool_embeddings shape', pool_embeddings.shape)
    np.save('../data/label_mean_embeddings.npy', mean_embeddings)
    np.save('../data/label_pool_embeddings.npy', pool_embeddings)
