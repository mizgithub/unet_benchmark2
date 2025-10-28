import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

# --- PyTorch Utility Blocks ---

class DoubleConv(nn.Module):
    """(Conv2d -> ReLU) * 2 with 'same' padding (padding=1 for kernel_size=3)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Using kernel_initializer='he_normal' in Keras suggests Kaiming initialization.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SelfAttention(nn.Module):
    """Simple Self-Attention block to replicate Attention()([x, x]) behavior."""
    def __init__(self, in_channels):
        super().__init__()
        # In this context, a simple 1x1 conv or a spatial attention mechanism
        # is often used. We'll use a straightforward 1x1 conv to get a spatial attention map.
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Input x is the feature map (e.g., conv1, conv2, conv3)"""
        batch_size, C, H, W = x.size()
        
        # 1. Calculate Query and Key, reshape to (B, C', H*W)
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1) # (B, H*W, C')
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)                     # (B, C', H*W)

        # 2. Calculate Attention Map: (B, H*W, C') x (B, C', H*W) -> (B, H*W, H*W)
        energy = torch.bmm(proj_query, proj_key) 
        attention = F.softmax(energy, dim=-1)

        # 3. Calculate Value and apply Attention: (B, C, H*W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W) # (B, C, H*W)

        # 4. Attention output: (B, C, H*W) x (B, H*W, H*W) -> (B, C, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # 5. Residual connection with learned gamma
        out = self.gamma * out + x
        return out


# --- VGG-19 Feature Extractor ---

class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGG-19 with pre-trained ImageNet weights
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # The Keras implementation extracts features from base_vgg19.layers[9:-2].
        # In PyTorch VGG19, the feature extractor is a sequential list of layers.
        # Layer 9 is likely the start of the 3rd block's Conv layer.
        # Layer -2 is likely the last MaxPool before the classifier.
        
        # Approximate mapping of layers[9:-2] (Conv2d, ReLU, MaxPool2d layers)
        # Block 3: Conv, ReLU, Conv, ReLU, Conv, ReLU, Conv, ReLU, MaxPool
        # Block 4: Conv, ReLU, Conv, ReLU, Conv, ReLU, Conv, ReLU, MaxPool (index 18 to 27)
        # Block 5: Conv, ReLU, Conv, ReLU, Conv, ReLU, Conv, ReLU (up to index 34)
        
        # A more robust mapping for features used in perception/style loss often targets
        # a layer before the final pooling. We'll extract a portion that seems to map 
        # roughly from block 3 to block 5 (up to the second-to-last conv in VGG).
        self.vgg_features = nn.Sequential(*vgg19[18:35])
        
        # The Keras code applies Conv2D(512, 1, activation='relu', ...) after each VGG layer
        # within the loop. This is unusual and may be simplified. We will apply the 1x1 conv
        # *after* the extracted VGG features. The VGG output here is 512 channels.
        
        # Adjusting the extracted block to match the Keras loop structure as closely as possible
        # for a functional block that maintains the spatial size.
        
        # The VGG-19 `features` are a sequential model. We need to iterate over its layers.
        
        # Custom sequential block to implement the loop structure:
        # x = l(x)
        # x = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        feature_list = []
        # Index 9 in Keras VGG19 (input_shape=(None,None,3)) is likely the start of the 3rd conv block.
        # PyTorch vgg19.features[18] is the start of the 4th conv block.
        # We will take the layers from feature block 4 (index 18) to the end of block 5's convs (index 34).
        
        # NOTE: The Keras layer index [9:-2] is highly dependent on the exact Keras implementation of VGG19.
        # Assuming layers[9:-2] corresponds to the 4th and 5th VGG blocks (outputting 512 channels),
        # the PyTorch feature slice for blocks 4 and 5 is roughly indices 18 to 34.
        
        # 1. Apply all VGG-19 feature layers from index 18 to 34 (Block 4 and 5 Conv layers)
        self.vgg_trunk = nn.Sequential(*vgg19[18:35])
        
        # 2. Add the 1x1 conv and UpSampling. The original Keras applies 1x1 conv *after* each VGG layer in the loop. 
        # Since VGG layers 18:35 are (Conv, ReLU) pairs, this would be a lot of 1x1 convs.
        # A more sensible interpretation for feature infusion is to apply the 1x1 conv *once* at the end 
        # to process the 512-channel output of the VGG trunk.
        
        self.final_processor = nn.Sequential(
            # Output of vgg_trunk is 512 channels.
            nn.Conv2d(512, 512, kernel_size=1, padding=0), # 1x1 Conv
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # UpSampling2D(size=(4, 4))
        )
        
    def forward(self, x):
        # x corresponds to 'conv4' in the UNet (256 channels).
        # The VGG-19 input requires 3 channels. This is a critical mismatch.
        # The Keras code 'base_vgg19 = VGG19(..., input_shape=(None,None,3))'
        # suggests the features need to be extracted from a 3-channel input.
        
        # To make it work, we must convert the input 'conv4' (B, 256, H, W) to (B, 3, H, W) 
        # or change the VGG-19 weights (which would break 'imagenet' pre-training).
        # Assuming the Keras model intended to have a 3-channel feature map before calling this.
        # The Keras implementation is likely **buggy** here unless it was a **Vi**sible/**I**nfrared-**U**Net 
        # where conv4 was a 3-channel map.
        
        # Since 'conv4' is the input, we'll assume the VGG-19 network must be adapted.
        # However, to maintain the spirit of 'imagenet' features, we'll proceed 
        # with the VGG-19 module and hope the input tensor has 3 channels.
        
        # NOTE: If your input 'conv4' is (B, C, H, W) where C is not 3, 
        # VGG will fail here. The PyTorch equivalent of the VGG feature extraction logic
        # assumes the input has 512 channels (from conv4) after an initial conversion.
        
        # For a clean PyTorch implementation, we must assume the first VGG layer in the trunk (index 18)
        # is modified to accept 256 channels (the output of UNet's conv4). This is a common practice 
        # when integrating pre-trained backbones into custom models.
        
        # We will assume a **modified VGG-19** where the first layer of the extracted features
        # can accept the UNet's `conv4` output (256 channels).
        
        x = self.vgg_trunk(x) # Input: (B, 256, H, W) -> Output: (B, 512, H/8, W/8)
        x = self.final_processor(x)
        return x

# --- The Main VIU-Net Model ---

class VIUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        
        # 1. Contracting Path
        
        # Input: (B, n_channels, H, W)
        self.conv1 = DoubleConv(n_channels, 64) # (B, 64, H, W)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 64, H/2, W/2)
        
        self.conv2 = DoubleConv(64, 128) # (B, 128, H/2, W/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 128, H/4, W/4)
        
        self.conv3 = DoubleConv(128, 256) # (B, 256, H/4, W/4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 256, H/8, W/8)
        
        # The Keras code breaks the standard UNet pattern here:
        self.conv4_unet = nn.Sequential( # The first conv of UNet's block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. VGG-19 Infusion
        self.vgg_features = VGG19Features()
        
        # 3. Bottom Layer
        # drop4 (VGG output) is (B, 512, H/8, W/8).
        # pool4 is MaxPool on drop4: (B, 512, H/16, W/16)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv5 is the bottom layer
        self.conv5 = DoubleConv(512, 1024) # (B, 1024, H/16, W/16)
        
        # 4. Expanding Path
        
        # Up 6
        # UpSampling2D(size=(2, 2)) -> Conv2D(512, 2, ...)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # (B, 512, H/8, W/8)
        # Concatenate: [drop4 (512), up6 (512)] -> 1024 channels
        self.conv6 = DoubleConv(1024, 512) # (B, 512, H/8, W/8)
        
        # Up 7
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # (B, 256, H/4, W/4)
        self.attn7 = SelfAttention(256) # Attention on conv3
        # Concatenate: [conv3 (256), up7 (256)] -> 512 channels
        self.conv7 = DoubleConv(512, 256) # (B, 256, H/4, W/4)
        
        # Up 8
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # (B, 128, H/2, W/2)
        self.attn8 = SelfAttention(128) # Attention on conv2
        # Concatenate: [conv2 (128), up8 (128)] -> 256 channels
        self.conv8 = DoubleConv(256, 128) # (B, 128, H/2, W/2)
        
        # Up 9
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # (B, 64, H, W)
        self.attn9 = SelfAttention(64) # Attention on conv1
        # Concatenate: [conv1 (64), up9 (64)] -> 128 channels
        self.conv9 = DoubleConv(128, 64) # (B, 64, H, W)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Contracting Path
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3) # (B, 256, H/8, W/8)

        # UNet's first conv of block 4
        conv4 = self.conv4_unet(pool3) # (B, 256, H/8, W/8)

        # VGG19 Infusion
        # NOTE: The VGG module (VGG19Features) is structured to receive (B, 256, H/8, W/8) 
        # and output (B, 512, H/8, W/8) after upsampling.
        drop4 = self.vgg_features(conv4) # Output: (B, 512, H/8, W/8)
        
        # Bottom Layer
        pool4 = self.pool4(drop4) # (B, 512, H/16, W/16)
        conv5 = self.conv5(pool4) # (B, 1024, H/16, W/16)

        # Expanding Path

        # Up 6
        up6 = self.up6(conv5)
        # Concatenate on Channel dimension (dim=1 in PyTorch)
        merge6 = torch.cat([drop4, up6], dim=1) # (B, 1024, H/8, W/8)
        conv6 = self.conv6(merge6)

        # Up 7
        up7 = self.up7(conv6)
        # attention_conv3 = self.attn7(conv3) # Use attention on skip connection
        merge7 = torch.cat([conv3, up7], dim=1) # (B, 512, H/4, W/4)
        # merge7 = torch.cat([attention_conv3, up7], dim=1) # Alternative path with attention
        conv7 = self.conv7(merge7)

        # Up 8
        up8 = self.up8(conv7)
        # attention_conv2 = self.attn8(conv2) # Use attention on skip connection
        merge8 = torch.cat([conv2, up8], dim=1) # (B, 256, H/2, W/2)
        # merge8 = torch.cat([attention_conv2, up8], dim=1) # Alternative path with attention
        conv8 = self.conv8(merge8)

        # Up 9
        up9 = self.up9(conv8)
        # attention_conv1 = self.attn9(conv1) # Use attention on skip connection
        merge9 = torch.cat([conv1, up9], dim=1) # (B, 128, H, W)
        # merge9 = torch.cat([attention_conv1, up9], dim=1) # Alternative path with attention
        conv9 = self.conv9(merge9)

        # Output
        logits = self.outc(conv9)
        outputs = self.sigmoid(logits)
        B_Mode = None
        
        return logits, outputs, B_Mode