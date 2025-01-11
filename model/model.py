import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from pytorchvideo.models.x3d import create_x3d

# 1. Feature Extractors
class FeatureExtractors(nn.Module):
    def __init__(self):
        super(FeatureExtractors, self).__init__()
        # X3D-S for Video
        self.video_x3d = torch.hub.load("facebookresearch/pytorchvideo", model="x3d_s", pretrained=True)
        self.video_x3d.blocks[-1] = nn.Identity()  # Remove classification head
        self.video_x3d_hybrid_pool = nn.AdaptiveAvgPool1d(5)
        self.video_x3d_linear_bottleneck = nn.Sequential(
            nn.Linear(24000, 2048),  # Placeholder; size will be adjusted dynamically
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # EfficientNet-B3 for Spectrogram
        self.spectrogram_efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.spectrogram_efficientnet.classifier = nn.Identity()  # Remove classification head
        self.eff_b3_linear_bottleneck = nn.Sequential(
            nn.Linear(1536, 2048),  # EfficientNet-b3 output 1536 features
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # PhoBERT for Text
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    def forward(self, video_frames, text, spectrogram):
        video_frames = video_frames.float()
        spectrogram = spectrogram.float()

        # Process video frames using batch * frames
        video_frames = video_frames.permute(0, 2, 1, 3, 4) # ONLY FOR X3D
        frame_features = self.video_x3d(video_frames)  # Output features from X3D
        batch_size, channels, frames, height, width = frame_features.shape
        frame_features = frame_features.permute(0, 2, 1, 3, 4).reshape(batch_size, frames, -1) # Flatten spatial dimensions (height, width, channels)
        frame_feature_size = frame_features.shape[-1] # Calculate the feature size dynamically
        frame_features = self.video_x3d_hybrid_pool(frame_features.permute(0, 2, 1))   # Pool over frames -> [batch, features, pooled_frames]
        frame_features = frame_features.permute(0, 2, 1).reshape(batch_size, -1)  # Flatten pooled frames and features
        if self.video_x3d_linear_bottleneck[0].in_features != frame_feature_size * 5:
            self.video_x3d_linear_bottleneck[0] = nn.Linear(frame_feature_size * 5, 2048).to("cuda")  # Explicitly move to device

        video_features = self.video_x3d_linear_bottleneck(frame_features)


        # Process spectrogram (static image)
        spectrogram_features = self.spectrogram_efficientnet(spectrogram)  # [Batch, 1536]
        spectrogram_features = self.eff_b3_linear_bottleneck(spectrogram_features) # [Batch, 1024]

        # Process text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(video_frames.device)

        # Extract text features
        text_features = self.phobert(**inputs).last_hidden_state[:, 0, :]  # CLS token -> [Batch, 768]

        return video_features, text_features, spectrogram_features

# 2. Tri-Attention Mechanism
import torch
import torch.nn as nn


class TriAttention(nn.Module):
    def __init__(self, dim_video, dim_text, dim_audio, common_dim = 1024):
        super(TriAttention, self).__init__()
        # Linear transformations to project all modalities to a common dimension
        self.Wv = nn.Linear(dim_video, common_dim)
        self.Wt = nn.Linear(dim_text, common_dim)
        self.Wa = nn.Linear(dim_audio, common_dim)

        # Learnable importance weights for each interaction
        self.alpha_vt = nn.Parameter(torch.tensor(1.0))  # Video-Text
        self.alpha_va = nn.Parameter(torch.tensor(1.0))  # Video-Audio
        self.alpha_ta = nn.Parameter(torch.tensor(0.5))  # Text-Audio

    def forward(self, video, text, audio):
        # Project all modalities to the common dimension
        video_proj = self.Wv(video)  # [batch_size, video_len, common_dim]
        text_proj = self.Wt(text)    # [batch_size, text_len, common_dim]
        audio_proj = self.Wa(audio)  # [batch_size, audio_len, common_dim]

        # Compute attention weights
        att_vt = torch.softmax(video_proj @ text_proj.transpose(-1, -2), dim=-1)  # Video-Text
        att_va = torch.softmax(video_proj @ audio_proj.transpose(-1, -2), dim=-1)  # Video-Audio
        att_ta = torch.softmax(text_proj @ audio_proj.transpose(-1, -2), dim=-1)  # Text-Audio

        # Modality interaction with importance weights
        f_v = video_proj + self.alpha_vt * (att_vt @ text_proj) + self.alpha_va * (att_va @ audio_proj)
        f_t = text_proj + self.alpha_vt * (att_vt.transpose(-1, -2) @ video_proj) + self.alpha_ta * (att_ta @ audio_proj)
        f_a = audio_proj + self.alpha_va * (att_va.transpose(-1, -2) @ video_proj) + self.alpha_ta * (att_ta.transpose(-1, -2) @ text_proj)

        # Combine features
        fused_features = torch.cat([f_v, f_t, f_a], dim=-1)  # Concatenate along the feature dimension
        return fused_features
    
class LuongTriAttention(nn.Module):
    def __init__(self, dim_video, dim_text, dim_audio, common_dim):
        super(LuongTriAttention, self).__init__()
        # Linear layers to project each modality to a common dimension
        self.Wa = nn.Linear(dim_audio, common_dim)  # For audio
        self.Wv = nn.Linear(dim_video, common_dim)  # For video
        self.Wt = nn.Linear(dim_text, common_dim)   # For text

        self.norm_a = nn.LayerNorm(common_dim)
        self.norm_v = nn.LayerNorm(common_dim)
        self.norm_t = nn.LayerNorm(common_dim)
        self.dropout = nn.Dropout(0.3)

        # Learnable importance weights initialized with specific values
        self.beta_iv = nn.Parameter(torch.tensor(2.0))  # Audio-Video importance
        self.beta_it = nn.Parameter(torch.tensor(0.8))  # Audio-Text importance
        self.beta_vt = nn.Parameter(torch.tensor(1.6))  # Video-Text importance

    def forward(self, video, text, audio):
        # Project modalities to the common dimension
        
        audio_proj = self.norm_a(self.dropout(self.Wa(audio))).unsqueeze(1)  # [Batch, 1, Common_Dim]
        video_proj = self.norm_v(self.dropout(self.Wv(video))).unsqueeze(1)  # [Batch, 1, Common_Dim]
        text_proj = self.norm_t(self.dropout(self.Wt(text))).unsqueeze(1)    # [Batch, 1, Common_Dim]
        
        # Compute attention scores using Luong's dot-product alignment
        att_iv = torch.softmax(torch.bmm(audio_proj, video_proj.transpose(1, 2)), dim=-1)  # [Batch, 1, 1]
        att_it = torch.softmax(torch.bmm(audio_proj, text_proj.transpose(1, 2)), dim=-1)  # [Batch, 1, 1]
        att_vt = torch.softmax(torch.bmm(video_proj, text_proj.transpose(1, 2)), dim=-1)  # [Batch, 1, 1]

        # Context vectors (weighted sum of projections)
        context_audio = self.beta_iv * torch.bmm(att_iv, video_proj) + self.beta_it * torch.bmm(att_it, text_proj)  # [Batch, 1, Common_Dim]
        context_video = self.beta_iv * torch.bmm(att_iv.transpose(1, 2), audio_proj) + self.beta_vt * torch.bmm(att_vt, text_proj)  # [Batch, 1, Common_Dim]
        context_text = self.beta_it * torch.bmm(att_it.transpose(1, 2), audio_proj) + self.beta_vt * torch.bmm(att_vt.transpose(1, 2), video_proj)  # [Batch, 1, Common_Dim]

        # Squeeze the sequence dimension for concatenation
        context_audio = context_audio.squeeze(1)  # [Batch, Common_Dim]
        context_video = context_video.squeeze(1)  # [Batch, Common_Dim]
        context_text = context_text.squeeze(1)    # [Batch, Common_Dim]

        # Concatenate the adjusted features from all contexts
        fused_features = torch.cat([context_audio, context_video, context_text], dim=-1)  # [Batch, 3 * Common_Dim]

        return fused_features

class EnhancedTriAttention(nn.Module):
    def __init__(self, dim_video, dim_text, dim_audio, common_dim, num_heads=8, dropout=0.3):
        # num_heads la so chia het cho toan bo dim
        super(EnhancedTriAttention, self).__init__()
        self.common_dim = common_dim

        # Linear layers to project modalities
        self.Wa = nn.Linear(dim_audio, common_dim)
        self.Wv = nn.Linear(dim_video, common_dim)
        self.Wt = nn.Linear(dim_text, common_dim)

        # Multi-head attention layers
        self.multihead_attn_audio_video = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout)
        self.multihead_attn_audio_text = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout)
        self.multihead_attn_video_text = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout)

        # Layer normalization and dropout
        self.norm_a = nn.LayerNorm(common_dim)
        self.norm_v = nn.LayerNorm(common_dim)
        self.norm_t = nn.LayerNorm(common_dim)
        self.dropout = nn.Dropout(dropout)

        # Modality weights
        self.weight_audio = nn.Parameter(torch.tensor(1.6))
        self.weight_video = nn.Parameter(torch.tensor(2.0))
        self.weight_text = nn.Parameter(torch.tensor(0.8))

        # Fusion fully connected layer with activation
        self.fc_fusion = nn.Linear(3 * common_dim, 3 * common_dim)
        self.activation = nn.ReLU()

    def forward(self, video, text, audio):
    # Project modalities to common dimension
            audio_proj = self.norm_a(self.dropout(self.Wa(audio)))  # [Batch, Common_Dim]
            video_proj = self.norm_v(self.dropout(self.Wv(video)))  # [Batch, Common_Dim]
            text_proj = self.norm_t(self.dropout(self.Wt(text)))    # [Batch, Common_Dim]

            # Add sequence dimension for multihead attention
            audio_proj = audio_proj.unsqueeze(0)  # [1, Batch, Common_Dim]
            video_proj = video_proj.unsqueeze(0)  # [1, Batch, Common_Dim]
            text_proj = text_proj.unsqueeze(0)    # [1, Batch, Common_Dim]

            # Multi-head attention
            att_iv, _ = self.multihead_attn_audio_video(audio_proj, video_proj, video_proj)
            att_it, _ = self.multihead_attn_audio_text(audio_proj, text_proj, text_proj)
            att_vt, _ = self.multihead_attn_video_text(video_proj, text_proj, text_proj)

            # Context vectors
            context_audio = self.weight_audio * (att_iv.mean(dim=0) + att_it.mean(dim=0))  # [Batch, Common_Dim]
            context_video = self.weight_video * (att_iv.mean(dim=0) + att_vt.mean(dim=0))  # [Batch, Common_Dim]
            context_text = self.weight_text * (att_it.mean(dim=0) + att_vt.mean(dim=0))    # [Batch, Common_Dim]

            # Concatenate and fuse
            fused_features = torch.cat([context_audio, context_video, context_text], dim=-1)  # [Batch, 3 * Common_Dim]
            fused_features = self.activation(self.fc_fusion(fused_features))  # Apply non-linearity

            return fused_features




# 3. Full Model
class TriModalModel(nn.Module):
    def __init__(self):
        super(TriModalModel, self).__init__()
        self.extractors = FeatureExtractors()
        self.attention = LuongTriAttention(dim_video=1024, dim_text=768, dim_audio=1024, common_dim=1024)
        # self.classifier = nn.Sequential(
        #     nn.Linear(1280 * 3, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(512, 5)  # 5 output classes
        # )
        self.classifier = nn.Sequential(
                    nn.Linear(1024 * 3, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.5),
                    nn.Linear(256, 5)  # Output classes (adjust as needed)
                )
    def forward(self, video_frames, text, spectrogram):
        video_features, text_features, audio_features = self.extractors(video_frames, text, spectrogram)
        fused_features = self.attention(video_features, text_features, audio_features)
        output = self.classifier(fused_features)
        return output