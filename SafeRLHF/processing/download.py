import pandas as pd

from SafeRLHF.config import ORIGINAL_DATA_FILE

splits = {'train': 'data/Alpaca-7B/train.jsonl',
          'test': 'data/Alpaca-7B/test.jsonl'}
train_df = pd.read_json(
    "hf://datasets/PKU-Alignment/PKU-SafeRLHF/" + splits["train"], lines=True)
test_df = pd.read_json(
    "hf://datasets/PKU-Alignment/PKU-SafeRLHF/" + splits["test"], lines=True)

original_df = pd.concat([train_df, test_df])
original_df.to_csv(ORIGINAL_DATA_FILE)
