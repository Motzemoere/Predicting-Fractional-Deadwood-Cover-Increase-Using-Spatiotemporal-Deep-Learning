"""
Lightweight multi-modal neural network for deadwood increase prediction.
Processes spatio-temporal, spatial, temporal, and auxiliary data through specialized embeddings.
"""

# Coming from the dataloader:

# SpatioTemporal:
    # deadwood_forest: torch.Size([128, 4, 33, 33, 3]) [Batch, Channels, Depth, Height, Time(yearly)] 
# Saptial:
    # terrain: torch.Size([128, 4, 33, 33]) [Batch, Channels, Depth, Height]
    # canopy: torch.Size([128, 2, 33, 33]) [Batch, Channels, Depth, Height]
# Temporal:
    # pixels_sentle: torch.Size([128, 12, 156]) [Batch, Bands, Time(weekly)]
    # era5: torch.Size([128, 11, 156]) [Batch, Bands, Time(weekly)]
# Auxiliary:
    # wc: torch.Size([128, 19]) [Batch, Features]
    # sg: torch.Size([128, 61]) [Batch, Features]
    # stand_age: torch.Size([128, 6]) [Batch, Features]

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeadwoodForestEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size=3, in_channels=4, history_years=3, img_size=33):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.history_years = history_years

        self.norm = nn.LayerNorm(embed_dim)
        
        # 1. Patch Projection
        # We treat each 3x3 patch across the 3 years as a source of information.
        # With img_size 33 and patch_size 3, we get 11x11 = 121 spatial tokens per year.
        self.num_patches_axis = img_size // patch_size
        self.num_patches = self.num_patches_axis ** 2
        
        # A 3D Conv stem is great for initial spatio-temporal feature extraction
        # It handles the 4 channels and small local spatial context simultaneously
        self.projection = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=(patch_size, patch_size, 1), 
            stride=(patch_size, patch_size, 1)
        )
        
        # 2. Positional Encodings
        # Spatial: 2D coordinates relative to the center pixel (16, 16)
        #self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Temporal: 1D encoding for the 3 years
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, history_years, embed_dim))
        
        # 3. Center Pixel Indicator
        # We add a learnable bias to the token that contains the center pixel
        # to explicitly "anchor" the model's focus.
        self.center_token_index = (self.num_patches_axis // 2) * self.num_patches_axis + (self.num_patches_axis // 2)
        self.center_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mask_projection = nn.Linear(1, embed_dim) # To embed the NaN mask info
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.center_bias, std=0.02)

    def forward(self, x, shared_spatial_pos):
        """
        Input x: [Batch, 4, 33, 33, 3] (Channels, H, W, Time)
        """
        B, C, H, W, T = x.shape
        
        # 1. Handle NaNs: Create mask and zero-fill
        # mask is 1 where data is valid, 0 where it is NaN
        mask = ~torch.isnan(x)
        x = torch.nan_to_num(x, nan=0.0)
        
        # 2. Extract Spatio-Temporal Tokens
        # projection output: [B, embed_dim, 11, 11, 3]
        tokens = self.projection(x)
        
        # Calculate token-level mask
        # If any pixel in the patch was a NaN, we want to flag the token
        # We downsample the mask using max_pool to match the tokens
        token_mask = F.max_pool3d(
            (~mask).any(dim=1, keepdim=True).float(), # [B, 1, 33, 33, 3]
            kernel_size=(self.patch_size, self.patch_size, 1),
            stride=(self.patch_size, self.patch_size, 1)
        ) # Result: [B, 1, 11, 11, 3]
        
        # Reorder to sequence format: [Batch, Time, Spatial_Patches, Embed_Dim]
        tokens = tokens.permute(0, 4, 2, 3, 1).flatten(2, 3) 
        # Result: [B, 3, 121, embed_dim]
        
        # 3. Apply Positional Encodings
        # Add spatial info (where is this patch in the 33x33?)
        tokens = tokens + shared_spatial_pos.unsqueeze(1)
        
        # Add temporal info (is this Year 1, 2, or 3?)
        tokens = tokens + self.temporal_pos_embed.unsqueeze(2)
        
        # 4. Center-Pixel Importance
        # Inject the center bias to the tokens representing the center patch across all years
        tokens[:, :, self.center_token_index, :] += self.center_bias
        
        # 5. Integrate Mask information
        # We add the embedded mask back into the tokens so the model "sees" the NaNs
        token_mask_seq = token_mask.permute(0, 4, 2, 3, 1).flatten(1, 3) # [B, 363, 1]
        mask_embed = self.mask_projection(token_mask_seq)
        
        # Flatten everything to a single sequence for the Transformer
        # [B, 3 * 121, embed_dim] -> [B, 363, embed_dim]
        tokens = tokens.flatten(1, 2)
        
        # Final tokens with mask awareness
        tokens = tokens + mask_embed
        
        # For the transformer attention mask:
        # 0 means "attend", 1 means "ignore"
        attn_mask = token_mask_seq.squeeze(-1).bool() 
        tokens = self.norm(tokens)
        return tokens, attn_mask

class SpatialStaticEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size=3, terrain_channels=5, canopy_channels=2, img_size=33):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.norm = nn.LayerNorm(embed_dim)
        
        # 1. Input fusion
        # terrain (4) + canopy (2) = 6 total static channels
        in_channels = terrain_channels + canopy_channels
        
        self.num_patches_axis = img_size // patch_size
        self.num_patches = self.num_patches_axis ** 2
        
        # 2. Spatial Projection (2D instead of 3D)
        # Summarizes 3x3 patches into a single vector
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 3. Positional Encoding & Center Focus
        #self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Anchor point for the center 3x3 patch
        self.center_token_index = (self.num_patches_axis // 2) * self.num_patches_axis + (self.num_patches_axis // 2)
        self.center_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mask_projection = nn.Linear(1, embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.center_bias, std=0.02)

    def forward(self, terrain, canopy, shared_spatial_pos):
        """
        terrain: [B, 4, 33, 33]
        canopy:  [B, 2, 33, 33]
        """
        # 1. Concatenate modalities
        # Result: [B, 6, 33, 33]
        x = torch.cat([terrain, canopy], dim=1)
        
        # 2. Handle NaNs
        mask = ~torch.isnan(x)
        x = torch.nan_to_num(x, nan=0.0)
        
        # 3. Patch Projection
        # [B, 6, 33, 33] -> [B, embed_dim, 11, 11]
        tokens = self.projection(x)
        
        # Calculate token-level mask (True where data is "bad")
        # We maxpool the "is_nan" boolean to see if any pixel in the 3x3 was invalid
        token_mask = F.max_pool2d(
            (~mask).any(dim=1, keepdim=True).float(), # Move .float() here
            kernel_size=self.patch_size,
            stride=self.patch_size
        ) # Result: [B, 1, 11, 11]
        
        # Reorder to sequence: [B, 121, embed_dim]
        tokens = tokens.flatten(2).transpose(1, 2)
        token_mask_seq = token_mask.flatten(1).unsqueeze(-1) # [B, 121, 1]
        
        # 4. Apply Spatial Context
        # Every static patch learns its position relative to the center
        tokens = tokens + shared_spatial_pos
        
        # Inject Center Bias
        tokens[:, self.center_token_index : self.center_token_index + 1, :] += self.center_bias
        
        # 5. Mask Awareness
        mask_embed = self.mask_projection(token_mask_seq)
        tokens = tokens + mask_embed
        
        # Boolean mask for Transformer attention (1 = ignore)
        attn_mask = token_mask_seq.squeeze(-1).bool()
        tokens = self.norm(tokens)
        return tokens, attn_mask

class TemporalWeeklyEmbedding(nn.Module):
    def __init__(self, embed_dim, sentle_channels=12, era5_channels=11, total_weeks=156):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_weeks = total_weeks
        self.norm = nn.LayerNorm(embed_dim)
        
        # 1. Temporal Projection (1D Conv)
        # We use a kernel_size of 4 to capture local "events" (approx. 1 month)
        # and a stride of 2 to reduce the sequence length to 78 tokens for speed.
        in_channels = sentle_channels + era5_channels
        self.projection = nn.Conv1d(
            in_channels, 
            embed_dim, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        
        self.num_tokens = total_weeks // 2
        
        # 2. Positional Encodings
        # Annual Trend: Which of the 3 years are we in? (0, 1, 2)
        self.year_embed = nn.Embedding(3, embed_dim)
        
        # Seasonality: Which week of the year is it? (1 to 52)
        # We use a Sine/Cosine encoding to represent the circular nature of seasons
        # (December is close to January)
        self.register_buffer("seasonal_pe", self._get_sinusoidal_encoding(total_weeks, embed_dim))

        # 3. Year Summary Tokens (Optional but powerful for "last year was dry" logic)
        # We add 3 "Year Summary" tokens that the model can use to aggregate annual stats
        self.year_summary_tokens = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
        self.mask_projection = nn.Linear(1, embed_dim)
        self.init_weights()

    def _get_sinusoidal_encoding(self, length, dim):
        # Standard Sine/Cosine Positional Encoding
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1)
        # We cycle every 52 weeks
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * (2 * math.pi / 52)) # Seasonal cycle
        pe[:, 1::2] = torch.cos(position * (2 * math.pi / 52))
        return pe.unsqueeze(0) # [1, 156, embed_dim]

    def init_weights(self):
        nn.init.trunc_normal_(self.year_summary_tokens, std=0.02)

    def forward(self, sentle, era5):
        """
        sentle: [B, 12, 156]
        era5:   [B, 11, 156]
        """
        # 1. Fusion & NaN handling
        x = torch.cat([sentle, era5], dim=1) # [B, 23, 156]
        mask = torch.isnan(x).any(dim=1, keepdim=True) # [B, 1, 156]
        x = torch.nan_to_num(x, nan=0.0)
        
        # 2. Local Trend Extraction
        # [B, 23, 156] -> [B, embed_dim, 78]
        tokens = self.projection(x)
        tokens = tokens.transpose(1, 2) # [B, 78, embed_dim]
        
        # 3. Apply Positional Context
        # Add Seasonality (Sine/Cosine) - Downsampled to match tokens
        seasonal_context = self.seasonal_pe[:, ::2, :] # Match stride 2
        tokens = tokens + seasonal_context
        
        # Add Annual Index (0, 1, 2)
        # Create year indices for the 78 tokens (0...0, 1...1, 2...2)
        year_indices = torch.arange(3, device=x.device).repeat_interleave(self.num_tokens // 3)
        tokens = tokens + self.year_embed(year_indices).unsqueeze(0)
        
        # 4. Integrate Year Summary Tokens
        # We prepend 3 tokens representing Year 1, Year 2, and Year 3
        # This allows the center pixel to attend to "Year 2" as a single concept
        B = x.shape[0]
        summary_tokens = self.year_summary_tokens.expand(B, -1, -1)
        tokens = torch.cat([summary_tokens, tokens], dim=1) # [B, 3 + 78, embed_dim]
        
        # 5. Mask Awareness
        token_mask = F.max_pool1d(
            mask.any(dim=1, keepdim=True).float(), # Move .float() here
            kernel_size=4, 
            stride=2, 
            padding=1
        ) # [B, 1, 78]
        # Add 3 False values for the summary tokens (they are always valid)
        summary_mask = torch.zeros(B, 1, 3, device=x.device)
        full_mask = torch.cat([summary_mask, token_mask], dim=2).squeeze(1).bool()
        tokens = self.norm(tokens)
        return tokens, full_mask
    
class FourierFeatureMapping(nn.Module):
    """
    Maps a continuous vector to a higher-dimensional Fourier space.
    This helps the model learn high-frequency variations in static data.
    """
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        # We create a fixed set of random frequencies
        # These are not trained, just used to project the data
        self.register_buffer("B", torch.randn(in_features, out_features // 2) * scale)

    def forward(self, x):
        # x shape: [Batch, in_features]
        # Project into frequency space
        projection = torch.matmul(x, self.B) # [Batch, out_features // 2]
        # Return [sin(v), cos(v)]
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)

class AuxiliaryStaticEmbedding(nn.Module):
    def __init__(self, embed_dim, wc_dim=19, sg_dim=61, sa_dim=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        
        # 1. Individual Fourier Mappers
        # We map each category to a temporary space before the final projection
        self.wc_fourier = FourierFeatureMapping(wc_dim, 128)
        self.sg_fourier = FourierFeatureMapping(sg_dim, 128)
        self.sa_fourier = FourierFeatureMapping(sa_dim, 64)
        
        # 2. Linear Projections to embed_dim
        # We will output 3 tokens: one for Climate (wc), one for Soil (sg), one for Stand (sa)
        self.wc_proj = nn.Linear(128, embed_dim)
        self.sg_proj = nn.Linear(128, embed_dim)
        self.sa_proj = nn.Linear(64, embed_dim)
        
        # Optional: A shared 'Static Category' embedding to help the model 
        # distinguish these tokens from the spatial/temporal ones
        self.category_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
    def forward(self, wc, sg, sa):
        """
        wc: [B, 19], sg: [B, 61], sa: [B, 6]
        """
        # 1. Handle NaNs in auxiliary data (common in WorldClim/SoilGrids)
        wc = torch.nan_to_num(wc, nan=0.0)
        sg = torch.nan_to_num(sg, nan=0.0)
        sa = torch.nan_to_num(sa, nan=0.0)
        
        # 2. Fourier Feature Mapping
        wc_feat = self.wc_fourier(wc) # [B, 128]
        sg_feat = self.sg_fourier(sg) # [B, 128]
        sa_feat = self.sa_fourier(sa) # [B, 64]
        
        # 3. Project to Transformer Dimension
        wc_token = self.wc_proj(wc_feat).unsqueeze(1) # [B, 1, embed_dim]
        sg_token = self.sg_proj(sg_feat).unsqueeze(1) # [B, 1, embed_dim]
        sa_token = self.sa_proj(sa_feat).unsqueeze(1) # [B, 1, embed_dim]
        
        # 4. Combine into 3 Global Tokens
        tokens = torch.cat([wc_token, sg_token, sa_token], dim=1) # [B, 3, embed_dim]
        
        # 5. Add category identity
        tokens = tokens + self.category_embed
        
        # These tokens are always valid, so no mask is needed (return False/Zeros)
        attn_mask = torch.zeros(wc.shape[0], 3, device=wc.device).bool()
        tokens = self.norm(tokens)
        return tokens, attn_mask
    
class MultimodalDeadwoodTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 num_heads=6, 
                 num_layers=4, 
                 dropout=0.1, 
                 mlp_ratio=4.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        num_patches = (33 // 3) ** 2

        # Shared Spatial Positional Embeddings
        self.shared_spatial_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.shared_spatial_pos, std=0.02)

        # 1. Specialized Embeddings
        self.forest_embed = DeadwoodForestEmbedding(embed_dim=embed_dim)
        self.static_embed = SpatialStaticEmbedding(embed_dim=embed_dim)
        self.temporal_embed = TemporalWeeklyEmbedding(embed_dim=embed_dim)
        self.aux_embed = AuxiliaryStaticEmbedding(embed_dim=embed_dim)

        # 2. Global CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Transformer Encoder Blocks
        # We use standard PyTorch Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Better stability for regression
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers, 
                                                         enable_nested_tensor=False)

        # 4. Final Regression Head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1) # Predicts deadwood_increase
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, deadwood_forest, terrain, canopy, pixels_sentle, era5, wc, sg, stand_age):
        B = deadwood_forest.shape[0]

        # --- STEP 1: Generate Modality Tokens & Masks ---
        # Forest tokens: [B, 363, D], mask: [B, 363]
        f_tokens, f_mask = self.forest_embed(deadwood_forest, self.shared_spatial_pos)
        
        # Static tokens: [B, 121, D], mask: [B, 121]
        s_tokens, s_mask = self.static_embed(terrain, canopy, self.shared_spatial_pos)
        
        # Temporal tokens: [B, 81, D], mask: [B, 81]
        t_tokens, t_mask = self.temporal_embed(pixels_sentle, era5)
        
        # Aux tokens: [B, 3, D], mask: [B, 3]
        a_tokens, a_mask = self.aux_embed(wc, sg, stand_age)

        # --- STEP 2: Prepend CLS Token & Concatenate ---
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, D]
        cls_mask = torch.zeros(B, 1, device=f_tokens.device).bool() # Always attend to CLS

        # Combine all tokens into a single sequence: [B, 1 + 363 + 121 + 81 + 3, D] -> [B, 569, D]
        tokens = torch.cat([cls_tokens, f_tokens, s_tokens, t_tokens, a_tokens], dim=1)
        
        # Combine all masks: [B, 569]
        full_mask = torch.cat([cls_mask, f_mask, s_mask, t_mask, a_mask], dim=1)

        # --- STEP 3: Contextualize via Transformer ---
        # src_key_padding_mask=True means the position is IGNORED (NaN values)
        encoded_sequence = self.transformer_encoder(tokens, src_key_padding_mask=full_mask)

        # --- STEP 4: Global Aggregation via CLS State ---
        # We take only the first token (the CLS token) which now contains 
        # information from all spatial, temporal, and auxiliary inputs.
        cls_representation = encoded_sequence[:, 0] # [B, D]

        # --- STEP 5: Regression Head ---
        prediction = self.regression_head(cls_representation) # [B, 1]

        return prediction
    