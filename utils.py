import random
import numpy as np
import torch
from torchvision import transforms
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    # Convert frame to a tensor and normalize to [0, 1]
    frame = torch.from_numpy(frame).float() / 255.0
    frame = transforms.ToPILImage()(frame.permute(2, 0, 1))
    
    # Resize and pad the frame
    transform = transforms.Compose([
        transforms.Resize(output_size, antialias=True),
        transforms.Pad((0, 0, max(0, output_size[1] - frame.size[1]), max(0, output_size[0] - frame.size[0]))),
        transforms.ToTensor()
    ])
    return transform(frame)

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded.
        output_size: Tuple specifying (height, width) of the output frame image.

    Returns:
        Formatted frame with padding of specified output size.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

class JSONDataset(Dataset):
    def __init__(self, json_path, n_frames=16, output_size_video=(224, 224), output_size_audio=(300,300), label_map=None):
        """
        Dataset class for loading data from JSON metadata.

        Args:
            json_path: Path to the JSON file containing dataset information.
            n_frames: Number of frames to extract from each video.
            output_size_video: Output video frame size (height, width).
            output_size_video: Output spectrogram size (height, width).
            label_map: Optional mapping from string labels to integers.
        """
        with open(json_path, "r", encoding="utf-8") as f:  # Specify utf-8 encoding
            self.data = json.load(f)
        self.n_frames = n_frames
        self.output_size_video = output_size_video
        self.output_size_audio = output_size_audio
        self.label_map = label_map if label_map else self._generate_label_map()

    def _generate_label_map(self):
        labels = set(item["label"] for item in self.data.values())
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[str(idx)]
        
        # Validate video_path and spec_path
        video_path = entry.get("video_path")
        spec_path = entry.get("spec_path")
        if not video_path or not spec_path:
            raise ValueError(f"Missing or invalid path for entry {idx}: video_path={video_path}, spec_path={spec_path}")

        # Convert paths to absolute paths
        video_path = Path(r'D:\Study\DS201\PROJECT\data') / video_path
        spec_path = Path(r'D:\Study\DS201\PROJECT\data') / spec_path

        text = entry.get("text", "")
        label = self.label_map[entry["label"]]

        video_frames = self._extract_frames(video_path)
        spec_image = self._process_spec(spec_path)
        # Normalize frames
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video_frames = torch.stack([
            normalize(torch.tensor(frame).permute(2, 0, 1)) for frame in video_frames
        ])
        spec_image = normalize(torch.tensor(spec_image).permute(2, 0, 1))

        return video_frames, text, spec_image, torch.tensor(label)

    def _extract_frames(self, file_path):
        """Extract frames uniformly across the video."""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {file_path}")

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count < self.n_frames:
            print(f"Warning: Video too short ({frame_count} frames). Will duplicate frames.")
            indices = np.linspace(0, frame_count - 1, self.n_frames, dtype=int)
        else:
            # Uniformly sample frame indices across the video
            indices = np.linspace(0, frame_count - 1, self.n_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(format_frames(frame, self.output_size_video))
            else:
                # Add black frame if reading fails
                frames.append(np.zeros((*self.output_size_video, 3), dtype=np.float32))
        
        cap.release()
        return np.array(frames, dtype=np.float32)
    def _process_spec(self, file_path):
        """Process a spectrogram as a static image."""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Unable to load spectrogram image: {file_path}")
        formatted_image = format_frames(image, self.output_size_audio)
        return formatted_image
    