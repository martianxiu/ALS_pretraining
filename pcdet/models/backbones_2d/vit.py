import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(ViTBackbone, self).__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.patch_size = self.model_cfg.get('PATCH_SIZE', 16)
        model_name = self.model_cfg.get('MODEL_NAME', 'google/vit-base-patch16-224')
        config = ViTConfig.from_pretrained(model_name)
        config.image_size = 468 # adjust to pointpillar setting

        # Initialize the Vision Transformer model without the pretrained weights
        self.vit = ViTModel(config)
        
        self.hidden_size = config.hidden_size
        self.output_channels = self.model_cfg.get('OUTPUT_CHANNELS', self.hidden_size)

        # Embedding layer to adapt the input channels to the hidden size expected by ViT
        # self.embedding = nn.Conv2d(input_channels, self.hidden_size, kernel_size=1)
        self.embedding = nn.Conv2d(input_channels, 3, kernel_size=1)

        # Projection layer to match the output channels to the desired size
        if self.output_channels != self.hidden_size:
            self.proj = nn.Conv2d(self.hidden_size, self.output_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        # Upsample layer to match the original resolution
        self.upsample = nn.Upsample(size=(468, 468), mode='bilinear', align_corners=False)

        self.num_bev_features = self.output_channels

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        # import pdb; pdb.set_trace()

        batch_size, num_channels, height, width = spatial_features.shape
        assert num_channels == self.input_channels, f"Expected input channels: {self.input_channels}, but got: {num_channels}"
        
        # Calculate the number of patches
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size

        # Apply the embedding layer to adapt the input channels
        x = self.embedding(spatial_features)  # (batch_size, hidden_size, height, width)

        # Run through Vision Transformer
        outputs = self.vit(x)

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state  # (batch_size, num_patches+ class token, hidden_size) e.g., torch.Size([2, 842, 768])

        # Optionally, remove the class token if not needed
        last_hidden_state = last_hidden_state[:, 1:, :]  # (batch_size, num_patches, hidden_size) 
        
        # Reshape to (batch_size, hidden_size, height, width)
        last_hidden_state = last_hidden_state.permute(0, 2, 1) # (batch_size, hidden_size, num_pathces) 
        last_hidden_state = last_hidden_state.view(batch_size, self.hidden_size,  num_patches_height, num_patches_width)

        # Project to the desired output channels if necessary
        last_hidden_state = self.proj(last_hidden_state)

        # Upsample to the original resolution
        upsampled_features = self.upsample(last_hidden_state)

        # Add the feature map to the data_dict
        data_dict['spatial_features_2d'] = upsampled_features
        
        # import pdb; pdb.set_trace()

        return data_dict

    def load_weights(self, pretrained_path):
        # Load pretrained weights if necessary
        self.vit.load_state_dict(torch.load(pretrained_path))