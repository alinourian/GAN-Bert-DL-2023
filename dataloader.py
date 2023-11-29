from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, AutoTokenizer
import torch
import csv
import pandas as pd



"""
pad zeros to tokenized inputs
"""
def zero_pad(data, max_len):
    l = data.shape[1]

    if l >= max_len:
        return data[0, :max_len]
    else:
        return F.pad(data, (0, max_len - l), value=0)[0]


"""
merge two csv files (Fake and True) and shuffle the data
"""
def merge_data(data_true, data_false, dest):
    df1 = pd.read_csv(data_true)
    df1['label'] = np.ones(len(df1), dtype=int)
    df2 = pd.read_csv(data_false)
    df2['label'] = np.zeros(len(df2), dtype=int)
    merged_df = pd.concat([df1, df2], axis=0)

    random_indices = np.random.permutation(merged_df.index)

    # Shuffle the DataFrame using the random indices
    shuffled_df = merged_df.loc[random_indices].reset_index(drop=True)
    shuffled_df.to_csv(dest, index=False)


"""
TextDataset class for loading data from csv file
@param file_name: path to csv file
@param tokenizer: tokenizer object
@param device: device to load data
@param data_info: dictionary containing information about data columns
@param max_len: maximum length of input tokens
@param loading_batch: number of lines to load at a time
"""
class TextDataset(Dataset):
    def __init__(self, file_name, tokenizer, device, data_info, max_len, loading_batch=10000):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.current_batch = -1
        self.num_lines = self.line_counter()
        self.data = []
        self.data_info = data_info
        self.loading_batch = loading_batch
        self.device = device
        self.max_len = max_len

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        index = self.check_index(index)
        data_txt = self.data[index][self.data_info['text']].rstrip()
        data_title = self.data[index][self.data_info['title']].rstrip()
        label = int(self.data[index][self.data_info['label']])
        data_txt = data_txt + ' [SEP]'
        data_title = data_title + ' [SEP]'
        # token_data = self.tokenizer.encode_plus(data, return_token_type_ids=True, truncation=True,
        #                                         max_length=max_len, return_attention_mask=True, padding='max_length',
        #                                         return_tensors='pt', add_special_tokens=True).data
        token_data_txt = self.tokenizer(data_txt, return_tensors="pt")
        token_data_title = self.tokenizer(data_title, return_tensors="pt")

        token_data_txt = {k: zero_pad(token_data_txt[k], self.max_len).to(self.device) for k in token_data_txt.keys()}
        token_data_title = {k: zero_pad(token_data_title[k], self.max_len).to(self.device) for k in token_data_title.keys()}
        one_hot = np.eye(2, dtype=int)[label]

        return token_data_title, token_data_txt, one_hot

    # load data from csv file
    def partial_data_loader(self):
        self.current_batch += 1
        start = self.current_batch * self.loading_batch
        if start < 1:
            start += 1
        end = start + self.loading_batch
        end = min(end, self.num_lines)

        with open(self.file_name, 'r') as f:
            if self.file_name.endswith('.csv'):
                csv_file = csv.reader(f)
                for i in range(start):
                    next(csv_file)

                self.data = []
                for i in range(start, end):
                    self.data.append(next(csv_file))
            else:
                self.data = f.readlines()[start:end]
        f.close()

    # check if index is in current batch
    def check_index(self, index):
        bias = (self.current_batch + 1) * self.loading_batch
        if index >= len(self.data) + bias:
            self.partial_data_loader()
            index -= self.current_batch * self.loading_batch
        return index

    # count number of lines in csv file
    def line_counter(self):

        with open(self.file_name, 'r') as f:
            if self.file_name.endswith('.csv'):
                num_lines = sum(1 for _ in csv.reader(f))
            else:
                num_lines = sum(1 for _ in f)
        f.close()

        return num_lines



"""
text_data_loader function for loading data from csv file
"""
def text_data_loader(data_file, tokenizer, device, data_info, max_len, batch_size=32, shuffle=True):
    dataset = TextDataset(data_file, tokenizer, device, data_info, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



device = torch.device('cpu')

# paths for data files
data_true = '/kaggle/input/fake-and-real-news-dataset/Fake.csv'
data_false = '/kaggle/input/fake-and-real-news-dataset/Fake.csv'
data_merge = '/kaggle/working/data/merged.csv'

# merge data files
merge_data(data_true, data_false, data_merge)

# information about data columns
data_info = {'title':0, 'text':1, 'subject':2, 'date':3, 'label':4}
# maximum length of input tokens (bert-base: 512)
max_len = 500

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_loader = text_data_loader(data_merge, tokenizer, device, data_info, max_len, shuffle=False)

# sample data
# token_data_title, token_data_txt, one_hot_label
token_data_title, token_data_txt, one_hot_label = train_loader.__iter__().__next__()

print(token_data_title['input_ids'])


"""
Want more data? First merge datasets in a .csv file, then use the dataloader.
There are plenty of them.
current dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""
