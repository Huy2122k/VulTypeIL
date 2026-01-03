import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from transformers import RobertaTokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from scipy.spatial import distance
import ast
import argparse

# Set parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 23
max_seq_length = 512
lr = 5e-5
num_epochs = 20
use_cuda = True
pretrained_model_path = "Salesforce/codet5-base"
ewc_lambda = 0.4  # EWC regularization term weight

# CWE list and mapping
CWE_LIST = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]
CWE_TO_INDEX = {cwe: idx for idx, cwe in enumerate(CWE_LIST)}

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for vulnerability type classification with continual learning.")
parser.add_argument('--data_dir', type=str, default='incremental_tasks_csv', help='Directory containing the CSV data files.')
parser.add_argument('--checkpoint_dir', type=str, default='.', help='Directory to save model checkpoints.')
parser.add_argument('--pretrained_model_path', type=str, default='Salesforce/codet5-base', help='Path to the pre-trained model.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs per task.')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA if available.')
args = parser.parse_args()

# Set parameters from args
batch_size = args.batch_size
num_class = 23
max_seq_length = 512
lr = args.lr
num_epochs = args.num_epochs
use_cuda = args.use_cuda
pretrained_model_path = args.pretrained_model_path
ewc_lambda = 0.4  # EWC regularization term weight

data_paths = [
    f'{args.data_dir}/task1_train.csv',
    f'{args.data_dir}/task2_train.csv',
    f'{args.data_dir}/task3_train.csv',
    f'{args.data_dir}/task4_train.csv',
    f'{args.data_dir}/task5_train.csv',
]

test_paths = [
    f'{args.data_dir}/task1_test.csv',
    f'{args.data_dir}/task2_test.csv',
    f'{args.data_dir}/task3_test.csv',
    f'{args.data_dir}/task4_test.csv',
    f'{args.data_dir}/task5_test.csv',
]

valid_paths = [
    f'{args.data_dir}/task1_valid.csv',
    f'{args.data_dir}/task2_valid.csv',
    f'{args.data_dir}/task3_valid.csv',
    f'{args.data_dir}/task4_valid.csv',
    f'{args.data_dir}/task5_valid.csv',
]

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# Define the classification model
class CodeT5Classifier(nn.Module):
    def __init__(self, model_name_or_path, num_classes):
        super(CodeT5Classifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# Read dataset and preprocess
def read_examples(filename):
    data = pd.read_csv(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    # Parse cwe_ids and map to index
    targets = []
    for cwe_str in data['cwe_ids']:
        try:
            cwe_list = ast.literal_eval(cwe_str)  # Parse string list
            cwe_id = cwe_list[0] if cwe_list else 'CWE-119'  # Take first CWE, default CWE-119
            target = CWE_TO_INDEX.get(cwe_id, 0)  # Map to index, default 0
        except:
            target = 0  # Fallback
        targets.append(target)
    texts = [f"{code[i]} [SEP] {desc[i]}" for i in range(len(code))]
    return texts, targets

def preprocess_data(texts, targets, tokenizer, max_seq_length):
    encodings = tokenizer(texts, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = torch.tensor(targets)
    return TensorDataset(input_ids, attention_mask, labels)

# Evaluate function
def evaluate(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    mcc = matthews_corrcoef(true_labels, preds)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    return acc, precision, recall, f1

# Hybrid Replay Strategy
def hybrid_replay(dataloader, model, device, num_samples=200):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, label_batch = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            features.append(logits.cpu().numpy())
            labels.extend(label_batch.cpu().tolist())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Calculate Mahalanobis distances
    mean_features = np.mean(features, axis=0)
    cov_matrix = np.cov(features, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
    distances = np.array([distance.mahalanobis(f, mean_features, cov_inv) for f in features])

    # Select uncertain samples
    tail_labels = {label for label, count in Counter(labels).items() if count < 0.05 * len(labels)}
    tail_indices = [i for i, label in enumerate(labels) if label in tail_labels]
    head_indices = [i for i in range(len(labels)) if i not in tail_indices]

    tail_selected = np.argsort(distances[tail_indices])[-num_samples // 2:]
    head_selected = np.argsort(distances[head_indices])[-num_samples // 2:]

    selected_indices = np.concatenate((np.array(tail_indices)[tail_selected], np.array(head_indices)[head_selected]))
    replay_features = features[selected_indices]
    replay_labels = labels[selected_indices]

    replay_dataset = TensorDataset(
        torch.tensor(replay_features, dtype=torch.float32),
        torch.tensor(replay_labels, dtype=torch.long)
    )
    return DataLoader(replay_dataset, batch_size=dataloader.batch_size, shuffle=True)

# Main loop
def main():
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
    model = CodeT5Classifier(pretrained_model_path, num_classes=num_class)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = None  # Add scheduler if needed

    for i in range(1, 6):
        print(f"----------------------- Task {i} ---------------------------")
        train_texts, train_targets = read_examples(data_paths[i - 1])
        train_dataset = preprocess_data(train_texts, train_targets, tokenizer, max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_texts, valid_targets = read_examples(valid_paths[i - 1])
        valid_dataset = preprocess_data(valid_texts, valid_targets, tokenizer, max_seq_length)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

        if i > 1:
            replay_dataloader = hybrid_replay(train_dataloader, model, device)
            for epoch in range(5):  # Replay for 5 epochs
                model.train()
                for batch in replay_dataloader:
                    features, labels = [b.to(device) for b in batch]
                    loss = nn.CrossEntropyLoss()(features, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Task {i} Epoch {epoch + 1}"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Task {i} Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

            print(f"Validation for Task {i} Epoch {epoch + 1}")
            evaluate(model, valid_dataloader, device)

        # Save the model
        torch.save(model.state_dict(), f"{args.checkpoint_dir}/best_model_task_{i}.pt")

        for j in range(1, 6):
            print(f"Evaluating on Task {j} Test Data")
            test_texts, test_targets = read_examples(test_paths[j - 1])
            test_dataset = preprocess_data(test_texts, test_targets, tokenizer, max_seq_length)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            evaluate(model, test_dataloader, device)

if __name__ == "__main__":
    main()
