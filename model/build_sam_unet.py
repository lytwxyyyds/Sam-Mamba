from model.sam_unet_model import SAM_UNET
from model.segment_anything.build_sam import sam_model_registry
from .config import config_dict
import torch


def build_res50_sam(need_ori_checkpoint, sam_unet_checkpoint):
    return _build_sam(
        type=50,
        ori_sam=sam_model_registry["vit_h"],
        ori_checkpoint=need_ori_checkpoint,
        sam_unet_checkpoint=sam_unet_checkpoint,
    )


def build_res34_sam(need_ori_checkpoint, sam_unet_checkpoint):
    return _build_sam(
        type=34,
        ori_sam=sam_model_registry['vit_b_256'],
        ori_checkpoint=need_ori_checkpoint,
        sam_unet_checkpoint=sam_unet_checkpoint,
    )

sam_unet_registry = {
    'res50_sam_unet': build_res50_sam,
    'res34_sam_unet': build_res34_sam,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_checkpoint = torch.load("F:/lyt/SamCD-main/SamCD-main/model/sam_vit_h_4b8939.pth", map_location=device)


def _build_sam(type,
               ori_sam, 
               ori_checkpoint: bool, 
               sam_unet_checkpoint: str):
    if sam_unet_checkpoint is not None and ori_checkpoint == False:
        sam_unet_checkpoint = torch.load(sam_unet_checkpoint)
        if 'model' in sam_unet_checkpoint.keys():
            sam_unet_checkpoint = sam_unet_checkpoint['model']
        model = SAM_UNET(resnet_type=type, ori_sam=ori_sam(None), is_resnet_pretrained_or_not=False)
        model.load_state_dict(sam_unet_checkpoint, map_location=device)
    elif ori_checkpoint == True: 
        model = SAM_UNET(resnet_type=type, ori_sam=ori_sam('F:\lyt\SamCD-main\SamCD-main\model\sam_vit_h_4b8939.pth').to(device), is_resnet_pretrained_or_not=True)
    else:
        model = SAM_UNET(resnet_type=type, ori_sam=ori_sam(None), is_resnet_pretrained_or_not=False)
    return model
