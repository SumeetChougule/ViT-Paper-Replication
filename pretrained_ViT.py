import torch
import requests
import torchvision
from torch import nn
from ViT_arc import data
from pathlib import Path
from torchinfo import summary
from src import data_setup, engine, utils, predictions

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

class_names = data.class_names

# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# 4. change the classifier head
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(
    device
)

# pretrained_vit

# # Print a summary using torchinfo (uncomment for actual output)
# summary(model=pretrained_vit,
#         input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )


# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()

train_dataloader_pretrained, test_dataloader_pretrained, class_names = (
    data_setup.create_dataloaders(
        train_dir=data.train_dir,
        test_dir=data.test_dir,
        transform=pretrained_vit_transforms,
        batch_size=32,
    )
)

# Train feature extractor ViT model

optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()

# Training

pretrained_vit_results = engine.train(
    model=pretrained_vit,
    train_dataloader=train_dataloader_pretrained,
    test_dataloader=test_dataloader_pretrained,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
)

# plot the loss curves
utils.plot_loss_curves(pretrained_vit_results)

# saving the pretrained model
utils.save_model(
    model=pretrained_vit,
    target_dir="models",
    model_name="pretrained_vit_feature_extractor_pizza_steak_sushi.pth",
)

# get the model size in bytes then convert to megabytes
pretrained_vit_model_size = Path(
    "models/pretrained_vit_feature_extractor_pizza_steak_sushi.pth"
).stat().st_size // (1024 * 1024)

# custom image path
custom_img_path = "custom.jpeg"

predictions.pred_and_plot_image(
    model=pretrained_vit, image_path=custom_img_path, class_names=class_names
)
