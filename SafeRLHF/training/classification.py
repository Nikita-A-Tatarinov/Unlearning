import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from config import LOG_FORMAT, LOG_LEVEL
from SafeRLHF.config import (
    CLS_TRAIN_TEST_DATA_DIR, CLS_SAFE_DATA_FILE_NAME,
    CLS_MODELS_DIR, CLS_MODELS_PLOTS_DIR,
    CLS_LOG_FILE)
from SafeRLHF.training.config import (
    VAL_RATIO, CLS_TARGET_NAME, CLS_EMBEDDING_DIM, CLS_HIDDEN_DIM,
    CLS_LR, CLS_MAX_LENGTH, NUM_EPOCHS)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=CLS_LOG_FILE)


class TextDataset(Dataset):
    def __init__(self, prompts, responses, labels, tokenizer, max_length=CLS_MAX_LENGTH):
        self.prompts = prompts
        self.responses = responses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(prompt + " " + response,
                                  truncation=True, padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=CLS_EMBEDDING_DIM,
                 hidden_dim=CLS_HIDDEN_DIM, output_dim=1, padding_idx=0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Shape: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(input_ids)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            packed_embedded = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        else:
            packed_embedded = embedded

        # hidden shape: (1, batch_size, hidden_dim)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output = self.fc(hidden[-1])  # Shape: (batch_size, output_dim)

        return self.sigmoid(output).squeeze(1)  # Shape: (batch_size,)


def load_data(train_test_data_dir: str, category: str, is_safe: bool, target_name: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dict = {}
    for group in ['train', 'test']:
        data_dict[group] = pd.read_csv(
            os.path.join(train_test_data_dir,
                         f'{category}_{group}.csv'))
        data_dict[group][target_name] = int(is_safe)
    return data_dict['train'], data_dict['test']


def get_dataset(df: pd.DataFrame, target_name: str, tokenizer):
    dataset = TextDataset(
        df['prompt'].tolist(),
        df['response'].tolist(),
        df[target_name].tolist(),
        tokenizer
    )
    return dataset


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(
            device)
        labels = batch['label'].float().to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(
                device)  # Move attention mask to device
            labels = batch['label'].to(device)

            # Get predictions from the model
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = (outputs > 0.5).int()

            # Move tensors to CPU and add to the lists for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds, average='binary')
    return f1


def plot_metrics(metrics: dict[list[float]], test_f1: float,
                 category: str, output_dir: str
                 ) -> None:
    epochs = range(1, len(metrics['loss']) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['loss'], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs for {category}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, category + '_loss.pdf'),
                format='pdf')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['f1'], label='F1 Score')
    plt.plot(epochs[-1], test_f1, marker='o', label='Test F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score over Epochs for {category}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, category + '_f1.pdf'),
                format='pdf')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    safe_train_data, safe_test_data = load_data(CLS_TRAIN_TEST_DATA_DIR,
                                                CLS_SAFE_DATA_FILE_NAME,
                                                True, CLS_TARGET_NAME)
    unsafe_categories = [f.split('_')[0] for f in os.listdir(CLS_TRAIN_TEST_DATA_DIR)
                         if f.endswith('.csv') and
                         f != f'{CLS_SAFE_DATA_FILE_NAME}_train.csv' and
                         f != f'{CLS_SAFE_DATA_FILE_NAME}_test.csv']
    unsafe_categories = list(set(unsafe_categories))

    for category in unsafe_categories:
        unsafe_train_data, unsafe_test_data = load_data(CLS_TRAIN_TEST_DATA_DIR,
                                                        category, False,
                                                        CLS_TARGET_NAME)
        train_df = pd.concat(
            [safe_train_data.sample(n=len(unsafe_train_data), random_state=42),
             unsafe_train_data]).reset_index(drop=True)
        test_df = pd.concat(
            [safe_test_data.sample(n=len(unsafe_test_data), random_state=42), unsafe_test_data]).reset_index(drop=True)
        train_df, val_df = train_test_split(train_df, test_size=VAL_RATIO,
                                            random_state=42,
                                            stratify=train_df[CLS_TARGET_NAME])

        train_dataset = get_dataset(
            train_df, CLS_TARGET_NAME, tokenizer)
        val_dataset = get_dataset(
            val_df, CLS_TARGET_NAME, tokenizer)
        test_dataset = get_dataset(
            test_df, CLS_TARGET_NAME, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = LSTMClassifier(tokenizer.vocab_size).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CLS_LR)

        metrics = {'loss': [], 'f1': []}

        logging.info(f'{category} category is being trained:')
        for epoch in range(NUM_EPOCHS):
            logging.info(f'\tEpoch {epoch + 1} out of {NUM_EPOCHS}')
            loss = train(model, train_loader, criterion, optimizer, device)
            logging.info(f'\tTrain loss: {loss}')
            f1 = evaluate(model, val_loader, device)
            logging.info(f'\tValidation f1: {f1}')
            metrics['loss'].append(loss)
            metrics['f1'].append(f1)

        logging.info('Testing')
        final_f1 = evaluate(model, test_loader, device)
        logging.info(f'Test f1: {final_f1}')

        plot_metrics(metrics, final_f1, category, CLS_MODELS_PLOTS_DIR)
        logging.info('Loss and f1 are plotted')

        torch.save(model.state_dict(),
                   os.path.join(CLS_MODELS_DIR,
                                f'{category}.pth'))
        logging.info('Model is saved')


if __name__ == '__main__':
    main()
