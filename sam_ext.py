## Modified from sam.build_sam.py

from typing import Tuple, Type
import torch
import numpy as np

from segment_anything import SamPredictor
from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, ImageEncoderViT
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn as nn

def build_sam_vit_h_no_encoder(checkpoint=None):
    return _build_sam_no_encoder(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

# build_sam = build_sam_vit_h

sam_model_registry_no_encoder = {
    "default": build_sam_vit_h_no_encoder,
    "vit_h": build_sam_vit_h_no_encoder,
}

class FakeImageEncoderViT(nn.Module):
    def __init__(self, img_size: int = 1024) -> None:
        super().__init__()
        self.img_size = img_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _build_sam_no_encoder(
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            encoder_global_attn_indexes,
            checkpoint=None,
        ):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=FakeImageEncoderViT(img_size=image_size),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        sam.load_state_dict(state_dict)
    return sam

class SamPredictorNoImgEncoder(SamPredictor):
    def __init__(
            self,
            sam_model: Sam,
            # image_encoder_img_size: int = 1024
            ) -> None:
        # super(SamPredictor, self).__init__()
        self.model = sam_model
        # self.transform = ResizeLongestSide(image_encoder_img_size)
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image_feature(self, img_features: np.ndarray, img_shape: Tuple[int, int]):
        self.features = torch.as_tensor(img_features, device=self.device) # .to(device=device)
        self.original_size = img_shape
        self.input_size = self.transform.get_preprocess_shape(img_shape[0], img_shape[1], self.model.image_encoder.img_size)
        self.is_image_set = True