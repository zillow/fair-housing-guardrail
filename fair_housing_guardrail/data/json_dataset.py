import random

from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_function(row, tokenizer, config):
    tensors = tokenizer(
        row[config["input_data"]["content_column"]].lower(),
        truncation=True,
        max_length=config["input_data"]["max_length"],
    )
    return tensors


class JsonDataset(Dataset):
    def __init__(self, df, label2id, tokenizer, config):
        self.data = list()
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            _sample = preprocess_function(row, tokenizer, config)
            _sample[config["input_data"]["label_column"]] = float(
                label2id[row[config["input_data"]["label_column"]]]
            )
            self.data.append(_sample)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
