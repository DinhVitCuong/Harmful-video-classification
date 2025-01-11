import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
import torch.nn as nn
import csv
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from utils import JSONDataset
from model.model import TriModalModel
from log_wandb import summarize_and_log_weights, summarize_and_log_gradients

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Environment Variables
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Hyperparameters and Constants
NUM_CLASSES = 5
N_FRAMES = 26
OUTPUT_SIZE_VIDEO = (160, 160)
OUTPUT_SIZE_AUDIO = (300, 300)
BATCH_SIZE = 4
LEARNING_RATE = 0.0005
ALLOWED_PATIENCE = 15
BEST_F1 = 0.0

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {DEVICE}")


# Dataset Paths
TRAIN_JSON_PATH = r"D:\Study\DS201\PROJECT\data\data_dict\train.json"
VAL_JSON_PATH = r"D:\Study\DS201\PROJECT\data\data_dict\val.json"
TEST_JSON_PATH = r"D:\Study\DS201\PROJECT\data\data_dict\test.json"

# Load Datasets and DataLoaders
print("LOADING DATA...")
train_dataset = JSONDataset(
    json_path=TRAIN_JSON_PATH,
    n_frames=N_FRAMES,
    output_size_video=OUTPUT_SIZE_VIDEO,
    output_size_audio=OUTPUT_SIZE_AUDIO
)
val_dataset = JSONDataset(
    json_path=VAL_JSON_PATH,
    n_frames=N_FRAMES,
    output_size_video=OUTPUT_SIZE_VIDEO,
    output_size_audio=OUTPUT_SIZE_AUDIO
)
test_dataset = JSONDataset(
    json_path=TEST_JSON_PATH,
    n_frames=N_FRAMES,
    output_size_video=OUTPUT_SIZE_VIDEO,
    output_size_audio=OUTPUT_SIZE_AUDIO
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DATA LOADED!")

# Initialize Model, Loss, and Optimizer
print("MODEL LOADING...")
model = TriModalModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("MODEL LOADED!")

# Evaluation Function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for video_frames, texts, spec_frames, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            try:
                video_frames, spec_frames, labels = (
                    video_frames.to(DEVICE), 
                    spec_frames.to(DEVICE), 
                    labels.to(DEVICE)
                )

                # Forward pass
                outputs = model(video_frames, texts, spec_frames)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Predictions and ground truth
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"Error during evaluation batch: {e}")

            finally:
                del video_frames, spec_frames, labels
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics Calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(NUM_CLASSES))
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, precision, recall, f1, avg_precision, avg_recall, avg_f1, all_labels, all_preds

# Test the Model
print("Testing the Model...")
model.load_state_dict(torch.load("best_trimodal_model.pth"))
test_loss, test_accuracy, test_precision, test_recall, test_f1, avg_precision, avg_recall, avg_f1, all_labels, all_preds = evaluate(model, test_loader, criterion)

# Print Test Metrics
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print("\nPer-Class Metrics:")
for i in range(NUM_CLASSES):
    print(f"Class {i}: Precision: {test_precision[i]:.4f}, Recall: {test_recall[i]:.4f}, F1-Score: {test_f1[i]:.4f}")
print("\nOverall Metrics:")
print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1:.4f}")
# Generate and Print Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))
