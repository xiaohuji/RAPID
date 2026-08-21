import sys
sys.path.append('../')  # adjust this to your local path
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config_3d
import torch

config_file = 'train/vit3d_highres'
pretrained_weights = r'D:\pyfile\RCC_Classify\rcc_code_CTMR\3dino_vit_weights.pth'  # adjust this to local path

cfg = load_and_merge_config_3d(config_file)
model = build_model_for_eval(cfg, pretrained_weights)

# the minimal preprocessing of the input image should be normalizing it to have values ranging between -1 and 1
# shape is batch size, channels, and spatial dims
example_img = torch.randn(1, 1, 112, 112, 112).cuda()

# for example:
# normalize 99.95% percentile to 1 and 0.05% percentile to -1, then clip to -1, 1
min_val = torch.quantile(example_img, 0.0005)
max_val = torch.quantile(example_img, 0.9995)
example_img = (example_img - min_val) / (max_val - min_val)
example_img = torch.clip(example_img * 2 - 1, -1, 1)

print(example_img.max(), example_img.min())

out = model(example_img)
