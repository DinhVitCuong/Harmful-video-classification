import torch
import torch.nn as nn
from model.model import TriModalModel
from utils import JSONDataset
from torch.utils.data import DataLoader
# Parameters
n_frames = 16  # Number of frames per video
output_size_video = (160, 160)  # Frame dimensions
output_size_audio = (300, 300)  # Frame dimensions
batch_size = 2
json_path = r"D:\Study\DS201\PROJECT\data\data_dict\test.json"
print("LOADING DATASET...")
dataset = JSONDataset(json_path=json_path, 
                      n_frames=n_frames, 
                      output_size_video=output_size_video, 
                      output_size_audio=output_size_audio)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("DATASET LOADED!")

for batch in dataloader:
    video_frames, texts, spec_frames, labels = batch
    print("Video Frames Shape:", video_frames.shape)
    print("VIDEO 1", video_frames[1][1])
    print("Text Samples:", texts)
    print("Spectrogram Shape:", spec_frames.shape)
    print("Labels Shape:", labels.shape)
    print("Labels:", labels)  
    break