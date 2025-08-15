from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

import utils_func.lovasz_loss as L
import imageio
from models.CDNet import CDMain
from utils_func.metrics import Evaluator
from data_cd.data_loader import ChangeDetectionDatset, make_data_loader
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
from configs.config import get_config
import numpy as np
import time
import os
import argparse
import sys
sys.path.append('./')


class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = CDMain(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
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
        )
        self.deep_model = self.deep_model.cuda()
        self.epoch = args.max_iters // args.batch_size

        self.change_map_saved_path = os.path.join(
            args.result_saved_path, args.dataset, args.model_type, 'change_map')

        if not os.path.exists(self.change_map_saved_path):
            os.makedirs(self.change_map_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            # Print the path of the loaded checkpoint
            print(f"Loading checkpoint from: {args.resume}")

            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()
    def canshu(self):
        x1 = torch.randn(1, 3, 256, 256).cuda()
        x2 = torch.randn(1, 3, 256, 256).cuda()

        res = self.deep_model(x1, x2)
        print(res[0].shape)

        from thop import profile

        # mmengine_flop_count(model, (3, 512, 512), show_table=True, show_arch=True)
        flops1, params1 = profile(self.deep_model, inputs=(x1, x2))
        print("flops=G", flops1)
        print("parms=M", params1)

    def infer(self):
        torch.cuda.empty_cache()
        dataset = ChangeDetectionDatset(
            self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(
            dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        self.evaluator.reset()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, names = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)

                image_name = names[0][0:-4] + f'.png'

                binary_change_map = np.squeeze(output_1)
                binary_change_map[binary_change_map == 1] = 255
                imageio.imwrite(os.path.join(
                    self.change_map_saved_path, image_name), binary_change_map.astype(np.uint8))

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        print('Inference stage is done!')

    def tezhen(self):
        # 准备测试输入
        img_path_1 = r"F:\lyt\levircd+\A\train_646_11.png"
        img_path_2 = r"F:\lyt\levircd+\B\train_646_11.png"

        # 定义图像预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量并归一化到 [0, 1]
        ])

        # 加载并预处理图像
        x1 = transform(Image.open(img_path_1).convert('RGB')).unsqueeze(0).cuda()  # [1,3,H,W]
        x2 = transform(Image.open(img_path_2).convert('RGB')).unsqueeze(0).cuda()

        # 定义存储特征图的字典
        features = {}

        # 注册钩子函数
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach().cpu()

            return hook

        # 假设我们要提取解码器第一层(dec1)和编码器最后一层的特征
        # 需要根据实际模型结构调整这些层名
        target_layers = {
            'decoder_first': self.deep_model.decoder.dec3,  # 解码器第一层
        }

        # 注册钩子
        hooks = []
        for name, layer in target_layers.items():
            hook = layer.register_forward_hook(get_features(name))
            hooks.append(hook)

        # 前向传播获取特征图
        with torch.no_grad():
            _ = self.deep_model(x1, x2)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 可视化和生成热力图
        for name, feat in features.items():
            print(f"{name}特征图形状: {feat.shape}")

            # 上采样特征图为 256x256
            upsampled_feat = F.interpolate(feat, size=(256, 256), mode='bilinear', align_corners=False)

            # 可视化热力图：生成单通道热力图
            # 选择第一个样本和一个通道来展示（例如通道0）
            feature_map = upsampled_feat[0, 0].numpy()  # 取第一个样本，第一个通道

            plt.figure(figsize=(10, 5))
            # 使用 viridis 色标来显示热力图
            plt.imshow(feature_map, cmap='viridis')
            plt.title(f"{name}热力图 (上采样到256x256) - 通道0")
            plt.colorbar()  # 显示颜色条
            plt.axis('off')  # 关闭坐标轴

            # 保存热力图
            save_path = os.path.join(self.change_map_saved_path, f"{name}_heatmap.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()  # 显示图像
            print(f"已保存{name}热力图到: {save_path}")

        print("热力图生成完成")



def main():
    parser = argparse.ArgumentParser(
        description="Training on SYSU/LEVIR-CD+/SYSU2-CD dataset")
    parser.add_argument(
        '--cfg', type=str, default=R'F:\lyt\SamCD-main\SamCD-main\configs\vssm.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SYSU2')
    parser.add_argument('--test_dataset_path', type=str,
                        default=r'F:\lyt\DSIFN\DSIFN')
    parser.add_argument('--test_data_list_path',
                        type=str, default=r'F:\lyt\DSIFN\DSIFN\list\test.txt')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='CD')
    parser.add_argument('--result_saved_path', type=str, default=r'F:\lyt\SamCD-main\SamCD-main\saved_model\DSIFN10')

    parser.add_argument('--resume', type=str, default=r'F:\lyt\SamCD-main\SamCD-main\saved_models\SYSU2\DSIFN-B\18000_model.pth')

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    infer = Inference(args)
    infer.infer()


if __name__ == "__main__":
    main()
