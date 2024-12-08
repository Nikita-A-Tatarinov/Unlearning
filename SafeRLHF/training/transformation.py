import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from config import LOG_FORMAT, LOG_LEVEL
from SafeRLHF.config import (
    TRANSFORMATION_TRAIN_TEST_DATA_DIR, TRANSFORMATION_MODELS_DIR,
    TRANSFORMATION_MODELS_PLOTS_DIR, TRANSFORMATION_LOG_FILE)
from SafeRLHF.training.config import (
    VAL_RATIO, NUM_EPOCHS, TRANSFORMATION_EMBEDDING_DIM, TRANSFORMATION_HIDDEN_DIM,
    TRANSFORMATION_MAX_LENGTH, TRANSFORMATION_NUM_HEADS, TRANSFORMATION_NUM_LAYERS,
    TRANSFORMATION_LR)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT,
                    filename=TRANSFORMATION_LOG_FILE)


class Seq2SeqDataset(Dataset):
    def __init__(self, unsafe_responses, safe_responses, tokenizer,
                 max_length=TRANSFORMATION_MAX_LENGTH):
        self.unsafe_responses = unsafe_responses
        self.safe_responses = safe_responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.unsafe_responses)

    def __getitem__(self, idx):
        unsafe_response = self.unsafe_responses[idx]
        safe_response = self.safe_responses[idx]

        src_encoding = self.tokenizer(unsafe_response, truncation=True,
                                      padding='max_length', max_length=self.max_length,
                                      return_tensors='pt')
        tgt_encoding = self.tokenizer(safe_response, truncation=True,
                                      padding='max_length', max_length=self.max_length,
                                      return_tensors='pt')

        return {
            'src_input_ids': src_encoding['input_ids'].flatten(),
            'src_attention_mask': src_encoding['attention_mask'].flatten(),
            'tgt_input_ids': tgt_encoding['input_ids'].flatten(),
            'tgt_attention_mask': tgt_encoding['attention_mask'].flatten(),
        }


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim=TRANSFORMATION_EMBEDDING_DIM,
                 num_heads=TRANSFORMATION_NUM_HEADS, ff_hidden_dim=TRANSFORMATION_HIDDEN_DIM,
                 num_layers=TRANSFORMATION_NUM_LAYERS, max_len=TRANSFORMATION_MAX_LENGTH):
        super(TransformerSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding with positional encoding
        src_embedded = self.embedding(
            src) + self.positional_encoding[:, :src.size(1), :]
        tgt_embedded = self.embedding(
            tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Transpose src_mask to (seq_length, batch_size) if necessary
        if src_mask is not None:
            # Transpose to match (seq_length, batch_size)
            src_mask = src_mask.T

        if tgt_mask is not None:
            # Transpose to match (seq_length, batch_size)
            tgt_mask = tgt_mask.T

        # Pass src_mask and tgt_mask to encoder and decoder
        memory = self.encoder(src_embedded, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_embedded, memory,
                              tgt_key_padding_mask=tgt_mask)

        # Final linear layer to map to vocab size
        return self.fc_out(output)


def load_data(train_test_data_dir: str, category: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dict = {}
    for group in ['train', 'test']:
        data_dict[group] = pd.read_csv(
            os.path.join(train_test_data_dir, f'{category}_{group}.csv'))
    return data_dict['train'], data_dict['test']


def train(model, dataloader, criterion, optimizer, device, pad_token_id):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src = batch['src_input_ids'].to(device)
        src_mask = batch['src_attention_mask'].to(device).bool()
        tgt = batch['tgt_input_ids'].to(device)
        tgt_mask = batch['tgt_attention_mask'].to(device).bool()

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # Shift target by one to align with output sequence length
        tgt = tgt[:, 1:]
        # Align output with target by removing last time step
        output = output[:, :-1, :]

        # Flatten output and target
        output = output.contiguous().view(-1, output.shape[-1])
        tgt = tgt.contiguous().view(-1)

        # Mask out padding tokens for loss calculation
        mask = tgt != pad_token_id
        output = output[mask]
        tgt = tgt[mask]

        # Calculate loss
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, pad_token_id):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src_input_ids'].to(device)
            src_mask = batch['src_attention_mask'].to(
                device).bool()  # No transpose
            tgt = batch['tgt_input_ids'].to(device)
            tgt_mask = batch['tgt_attention_mask'].to(
                device).bool()  # No transpose

            # Forward pass without slicing
            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

            # Shift target by one to align with output sequence length
            tgt = tgt[:, 1:]
            # Align output with target by removing last time step
            output = output[:, :-1, :]

            # Flatten output and target
            output = output.contiguous().view(-1, output.shape[-1])
            tgt = tgt.contiguous().view(-1)

            # Mask out padding tokens for loss calculation
            mask = tgt != pad_token_id
            output = output[mask]
            tgt = tgt[mask]

            # Calculate loss
            loss = criterion(output, tgt)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    unsafe_categories = [f.split('_')[0] for f in os.listdir(TRANSFORMATION_TRAIN_TEST_DATA_DIR)
                         if f.endswith('.csv')]
    unsafe_categories = list(set(unsafe_categories))

    for category in unsafe_categories:
        train_df, test_df = load_data(
            TRANSFORMATION_TRAIN_TEST_DATA_DIR, category)
        train_df, val_df = train_test_split(
            train_df, test_size=VAL_RATIO, random_state=42)

        train_dataset = Seq2SeqDataset(
            train_df['unsafe_response'].tolist(), train_df['safe_response'].tolist(), tokenizer)
        val_dataset = Seq2SeqDataset(
            val_df['unsafe_response'].tolist(), val_df['safe_response'].tolist(), tokenizer)
        test_dataset = Seq2SeqDataset(
            test_df['unsafe_response'].tolist(), test_df['safe_response'].tolist(), tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = TransformerSeq2Seq(tokenizer.vocab_size).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=TRANSFORMATION_LR)

        metrics = {'train_loss': [], 'val_loss': []}

        logging.info(f'{category} category is being trained:')
        for epoch in range(NUM_EPOCHS):
            logging.info(f'\tEpoch {epoch + 1} out of {NUM_EPOCHS}')
            train_loss = train(model, train_loader, criterion, optimizer,
                               device, tokenizer.pad_token_id)
            logging.info(f'\tTrain loss: {train_loss:.4f}')
            val_loss = evaluate(model, val_loader, criterion,
                                device, tokenizer.pad_token_id)
            logging.info(f'\tVal loss: {val_loss:.4f}')
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)

        logging.info('Testing')
        test_loss = evaluate(model, test_loader, criterion,
                             device, tokenizer.pad_token_id)
        logging.info(f'\tTest loss: {test_loss:.4f}')

        plt.plot(metrics['train_loss'], label='Train Loss')
        plt.plot(metrics['val_loss'], label='Val Loss')
        plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
        plt.legend()
        plt.savefig(os.path.join(TRANSFORMATION_MODELS_PLOTS_DIR,
                                 f'{category}_loss.pdf'),
                    format='pdf')
        plt.close()

        torch.save(model.state_dict(), os.path.join(
            TRANSFORMATION_MODELS_DIR, f'{category}.pth'))
        logging.info('Model is saved')


if __name__ == '__main__':
    main()
