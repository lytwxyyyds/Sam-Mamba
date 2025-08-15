import sys

from fvcore.nn import FlopCountAnalysis

sys.path.append('./')
from models.CDDecoder import CDDecoder
from models.vmamba import LayerNorm2d
import torch.nn as nn
import torch
import torch.nn.functional as F
from configs.config import get_config
import argparse
from sam2.build_sam import build_sam2
# from model.sam_unet_model import SAM_UNET
# from model.segment_anything.build_sam import sam_model_registry
# from model.build_sam_unet import _build_sam
# from model.build_sam_unet import sam_unet_registry,build_res50_sam
import torch


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
class CDMain(nn.Module):
    def __init__(self, **kwargs):
        super(CDMain, self).__init__()

        model_cfg = "F:\lyt\SamCD-main\SamCD-main\sam2_configs\sam2_hiera_b+.yaml"
        checkpoint_path = "F:\lyt\SamCD-main\SamCD-main\sam2_hiera_base_plus.pt"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder_sam = model.image_encoder.trunk

        for param in self.encoder_sam.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder_sam.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder_sam.blocks = nn.Sequential(
            *blocks
        )

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(
            kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(
            kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(
            kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = CDDecoder(
            encoder_dims=[112, 224, 448, 896],
            channel_first=False,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(
            in_channels=128, out_channels=2, kernel_size=1)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):

        pre_features = self.encoder_sam(pre_data)
        post_features = self.encoder_sam(post_data)

        output = self.decoder(pre_features, post_features)

        output = self.main_clf(output)
        output = F.interpolate(
            output, size=pre_data.size()[-2:], mode='bilinear')
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument('--cfg', type=str,
                        default=r'F:\lyt\SamCD-main\SamCD-main\configs\vssm.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str,default='F:\lyt\SamCD-main\SamCD-main\sam2_hiera_large.pt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str,
                        default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    config = get_config(args)
    model = CDMain(

        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 15, 2],
        dims=128,
        # ===================
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK ==
                               "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        # ===================
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        # ===================
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    ).cuda()

    x1 = torch.randn(1, 3, 1024, 1024).cuda()
    x2 = torch.randn(1, 3, 1024, 1024).cuda()
    # 假设你已经选择了一个设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移到相同的设备上
    model = model.to(device)

    # 确保输入数据也在同一设备上
    x1, x2 = x1.to(device), x2.to(device)

    res = model(x1, x2)

    print(res[0].shape)

    from thop import profile

    # mmengine_flop_count(model, (3, 512, 512), show_table=True, show_arch=True)
    flops1, params1 = profile(model, inputs=(x1, x2))
    print("flops=G", flops1)
    print("parms=M", params1)

from fvcore.nn import FlopCountAnalysis
import torch
