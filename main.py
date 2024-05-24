import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torchinfo import summary
from src import data_setup, engine, utils

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

image_path = utils.download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi",
)

# dir paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# create image size (from Table 3 of ViT paper)
IMG_SIZE = 224

# transform pipeline
manual_transforms = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

# Set the batch size
BATCH_SIZE = (
    32  # this is lower than the ViT paper but it's because we're starting small
)

# Create data loaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE,
)

# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# get a single image from the batch
image, label = image_batch[0], label_batch[0]
image.shape

# plot with matplotlib
plt.imshow(
    image.permute(1, 2, 0)
)  # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.title(class_names[label])
plt.axis(False)


# example values
height = 224  # H
width = 224  # W
color_channels = 3  # C
patch_size = 16  # P

# N (number of patchs)
N = int((height * width) / patch_size**2)

# input shape
embedding_layer_input_shape = (height, width, color_channels)

# output shape
embedding_layer_output_shape = (N, (patch_size**2) * color_channels)


# view single image
plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
plt.axis(False)

# Change image shape to be compatible with matplotlib (color_channels, height, width) -> (height, width, color_channels)
image_permuted = image.permute(1, 2, 0)

# index to plot the top row of patched pixels
patch_size = 16
plt.figure(figsize=(patch_size, patch_size))
plt.imshow(image_permuted[:patch_size, :, :])


# Setup hyperparameters and make sure img_size and patch_size are compatible
img_size = 224
patch_size = 16
num_patches = img_size / patch_size
assert img_size % patch_size == 0, "Image size must be divisible by patch size"


# create a series of subplots
fig, axs = plt.subplots(
    nrows=1,
    ncols=img_size // patch_size,
    figsize=(num_patches, num_patches),
    sharex=True,
    sharey=True,
)

# iterate through number of patchs in the top row
for i, patch in enumerate(range(0, img_size, patch_size)):
    axs[i].imshow(image_permuted[:patch_size, patch : patch + patch_size, :])
    axs[i].set_xlabel(i + 1)
    axs[i].set_xticks([])
    axs[i].set_yticks([])


# patches of the whole image
fig, axs = plt.subplots(
    nrows=img_size // patch_size,
    ncols=img_size // patch_size,
    figsize=(num_patches, num_patches),
    sharex=True,
    sharey=True,
)

for i, patch_heigth in enumerate(range(0, img_size, patch_size)):
    for j, patch_width in enumerate(range(0, img_size, patch_size)):
        # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
        axs[i, j].imshow(
            image_permuted[
                patch_heigth : patch_heigth + patch_size,
                patch_width : patch_width + patch_size,
                :,
            ]
        )
        # Set up label information, remove the ticks for clarity and set labels to outside
        axs[i, j].set_ylabel(
            i + 1,
            rotation="horizontal",
            horizontalalignment="right",
            verticalalignment="center",
        )
        axs[i, j].set_xlabel(j + 1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].label_outer()

# set a super title
fig.suptitle(f"{class_names[label]}-> Patchified", fontsize=16)
plt.show()

# set the patch size
patch_size = 16

# Create the Conv2d layer with hyperparameters from the ViT paper
conv2d = nn.Conv2d(
    in_channels=3,
    out_channels=768,  # from Table 1: Hidden size D, this is the embedding size
    kernel_size=patch_size,
    stride=patch_size,
    padding=0,
)

# view single image
plt.imshow(image.permute(1, 2, 0))  # adjust for matplot
plt.title(class_names[label])
plt.axis(False)


# pass the image through conv layer
image_out_of_conv = conv2d(image.unsqueeze(0))
image_out_of_conv.shape

# Plot random 5 convolutional feature maps
import random

random_indexes = random.sample(range(0, 768), k=10)

# create plot
fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(12, 12))

# plot random image feature maps
for i, idx in enumerate(random_indexes):
    image_conv_feature_map = image_out_of_conv[:, idx, :, :]
    axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy())
    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

single_feature_map = image_out_of_conv[:, 0, :, :]
single_feature_map.requires_grad


# create the flatten layer
flatten = nn.Flatten(start_dim=2, end_dim=3)


# 1. View single image
plt.imshow(image.permute(1, 2, 0))  # adjust for matplotlib
plt.title(class_names[label])
plt.axis(False)
image.shape

# 2. Turn image into feature maps
image_out_of_conv = conv2d(
    image.unsqueeze(0)
)  # add batch dimension to avoid shape errors
image_out_of_conv.shape

# 3. flatten the feature maps
image_out_of_conv_flattened = flatten(image_out_of_conv)
image_out_of_conv_flattened.shape

# Get flattened image patch embeddings in right shape
image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute(
    0, 2, 1
)  # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
image_out_of_conv_flattened_reshaped.shape


# Get a single flattened feature map
single_flattened_feature_map = image_out_of_conv_flattened_reshaped[:, :, 0]

# plot the flattened feature map visually
plt.figure(figsize=(22, 22))
plt.imshow(single_flattened_feature_map.detach().numpy())
plt.title(f"Flattened feature map shape: {single_flattened_feature_map.shape}")
plt.axis(False)

# See the flattened feature map as a tensor
single_flattened_feature_map
single_flattened_feature_map.requires_grad
single_flattened_feature_map.shape


# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()

        # 3. create a layer to turn an image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 4. create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    # 5. define the forward method
    def forward(self, x):
        # create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert (
            image_resolution % patch_size == 0
        ), f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


torch.manual_seed(42)
torch.mps.manual_seed(42)
# create an instance of patch embedding
patchify = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)
image.unsqueeze(0).shape

patch_embedded_image = patchify(image.unsqueeze(0))
patch_embedded_image.shape


# Create random input sizes
random_input_image = (1, 3, 224, 224)
random_input_image_error = (
    1,
    3,
    250,
    250,
)  # will error because image size is incompatible with patch_size

# # Get a summary of the input and outputs of PatchEmbedding (uncomment for full output)
summary(
    PatchEmbedding(),
    input_size=random_input_image,  # try swapping this for "random_input_image_error"
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# Get the batch size and embedding dimension
batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]

# Create the class token embedding as a learnable parameter that shares the same size as the embedding dimension (D)
class_token = nn.Parameter(
    torch.ones(
        batch_size, 1, embedding_dimension
    ),  # [batch_size, number_of_tokens, embedding_dimension]
    requires_grad=True,
)
class_token[:, :, :10]

# Add the class token embedding to the front of the patch embedding
patch_embedded_image_with_class_embedding = torch.cat(
    (class_token, patch_embedded_image), dim=1
)
patch_embedded_image_with_class_embedding.shape

# positional embedding

# calculate N
number_of_patches = int((height * width) / patch_size**2)

# get embedding dim
embedding_dimension = patch_embedded_image_with_class_embedding.shape[-1]

# create a learnable 1D position embedding
position_embedding = nn.Parameter(
    torch.ones(batch_size, number_of_patches + 1, embedding_dimension),
    requires_grad=True,
)
position_embedding.shape


# Add the position embedding to the patch and class token embedding
patch_and_position_embedding = (
    patch_embedded_image_with_class_embedding + position_embedding
)
patch_and_position_embedding.shape


# 1. Set patch size
patch_size = 16

# 2. Print shape of original image tensor and get the image dimensions
print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

# 3. get the image tensor and add batch dimension
x = image.unsqueeze(0)
print(f"Input image with batch dimension shape: {x.shape}")

# 4. create patch embedding layer
patch_embedding_layer = PatchEmbedding(
    in_channels=3, patch_size=patch_size, embedding_dim=768
)

# 5. pass image through patch embedding layer
patch_embedding = patch_embedding_layer(x)
print(f"Patch embedding shape: {patch_embedding.shape}")

# 6. create class token embedding
batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
class_token = nn.Parameter(
    torch.ones(batch_size, 1, embedding_dimension), requires_grad=True
)
print(f"Class token embedding shape: {class_token.shape}")

class_token.shape
class_token.expand(batch_size, -1, -1).shape

# 7. prepend to patch embedding
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

# 8. position embedding
number_of_patches = int((height * width) / patch_size**2)
position_embedding = nn.Parameter(
    torch.ones(batch_size, number_of_patches + 1, embedding_dimension),
    requires_grad=True,
)

# 9. adding class and position embedding
patch_and_position_embedding = patch_embedding_class_token + position_embedding
print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")


# Equation 2 of ViT
# 1. Create a class that inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short)"""

    # 2. Initialize the class with hyperparameters from table 1
    def __init__(
        self,
        embedding_dim: int = 768,  # hidden size D from table 1 for ViT-Base
        num_heads: int = 12,  # Heads from table 1 for ViT base
        attn_dropout: float = 0,  # doesn't look like the paper uses any dropout in MSA blocks
    ):
        super().__init__()

        # 3. create Norm Layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. create teh multi.head attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    # 5. create a forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x,  # query embeddings
            key=x,  # key embeddings
            value=x,  # value embeddings
            need_weights=False,
        )

        return attn_output


# create an instance of MSABlock
multihead_self_attention_block = MultiheadSelfAttentionBlock(
    embedding_dim=768, num_heads=12
)

patched_image_through_msa_block = multihead_self_attention_block(
    patch_and_position_embedding
)

patched_image_through_msa_block.shape
patched_image_through_msa_block[:, 0].shape


# Equation 3 of ViT
# MLP


# 1. create a class that inherits from nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ('MLP block' for short)."""

    # 2. Initialize the class with hyperparameters from tables 1 and 3
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
        mlp_size: int = 3072,  # MLP Size from Table 1 for ViT-Base
        dropout: float = 0,  # Dropout from table 3 for ViT base
    ):
        super().__init__()

        # 3. Create layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    # 5. forword method
    def forward(self, x):
        return self.mlp(self.layer_norm(x))


# an instance of MLPBlock
mlp_block = MLPBlock()

patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
patched_image_through_mlp_block.shape


# Transformer Encoder


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    # 2. Initialize the class with hyper params from tables 1 and 3
    def __init__(
        self,
        embedding_dim: int = 768,  # hidden size D
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0,
    ):
        super().__init__()

        # 3. create MSA block (eq 2)
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        # 4. create MLP block (eq 3)
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    # create a forward method
    def forward(self, x):

        # 6. Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


transformer_encoder_block = TransformerEncoderBlock()

# Print an input and output summary of our Transformer Encoder
summary(
    model=transformer_encoder_block,
    input_size=(1, 197, 768),  # (batch_size, num_patches, embedding_dimension)
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# Create the same as above with torch.nn.TransformerEncoderLayer()
torch_transformer_encoder_layer = nn.TransformerEncoderLayer(
    d_model=768,  # Hidden size D from Table 1 for ViT-Base
    nhead=12,  # Heads from Table 1 for ViT-Base
    dim_feedforward=3072,  # MLP size from Table 1 for ViT-Base
    dropout=0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
    activation="gelu",  # GELU non-linear activation
    batch_first=True,  # Do our batches come first?
    norm_first=True,
)  # Normalize first or after MSA/MLP layers?

torch_transformer_encoder_layer


# Get the output of PyTorch's version of the Transformer Encoder (uncomment for full output)
summary(
    model=torch_transformer_encoder_layer,
    input_size=(1, 197, 768),  # (batch_size, num_patches, embedding_dimension)
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)


# Example of creating the class embedding and expanding over a batch dimension
batch_size = 32
class_token_embedding_single = nn.Parameter(
    data=torch.randn(1, 1, 768)
)  # create a single learnable class token
class_token_embedding_expanded = class_token_embedding_single.expand(
    batch_size, -1, -1
)  # expand the single learnable class token across the batch dimension, "-1" means to "infer the dimension"

# Print out the change in shapes
print(f"Shape of class token embedding single: {class_token_embedding_single.shape}")
print(
    f"Shape of class token embedding expanded: {class_token_embedding_expanded.shape}"
)
