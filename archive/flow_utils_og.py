import pathlib
import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class RAFT(nn.Module):
    def __init__(self, model='things', num_iters=5, dropout=0):
        super(RAFT, self).__init__()

        from flow_models.raft import raft

        if model == 'things':
            model = 'raft-things.pth'
        else:
            raise NotImplementedError

        # Get location of checkpoints
        raft_dir = pathlib.Path(__file__).parent.absolute()/'flow_models'/'raft'
        checkpoint_path = raft_dir / model

        # Emulate arguments
        args = argparse.Namespace()
        args.model = checkpoint_path
        args.small = False
        args.mixed_precision = True
        args.alternate_corr = False
        args.dropout = dropout

        # Initialize RAFT model (WITHOUT DataParallel)
        self.flowNet = raft.RAFT(args)

        # Load checkpoint and modify keys if necessary
        checkpoint = torch.load(args.model, map_location='cpu')

        # If the checkpoint contains "state_dict", extract it
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # Fix key mismatches: If the checkpoint keys start with "module.", remove it
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_checkpoint[new_key] = value

        # Load the modified state_dict
        self.flowNet.load_state_dict(new_checkpoint, strict=False)

        self.num_iters = num_iters

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''
        # Normalize to [0, 255]
        im1 = im1 * 255
        im2 = im2 * 255

        # Estimate flow
        flow_low, flow_up = self.flowNet(im1, im2, iters=self.num_iters, test_mode=True)

        return flow_up
    
class ARFlow(nn.Module):
    def __init__(self):
        super(ARFlow, self).__init__()
        
        from flow_models.ARFlow.models.pwclite import PWCLite
        from easydict import EasyDict
        from utils.torch_utils import restore_model

        chkpt_path = pathlib.Path(__file__).parent.absolute() / 'flow_models/ARFlow/checkpoints/KITTI15/pwclite_ar.tar'

        model_cfg = EasyDict({'upsample': True, 'n_frames': 2, 'reduce_dense': True})
        flowNet = PWCLite(model_cfg)
        flowNet = restore_model(flowNet, chkpt_path)
        self.flowNet = flowNet

    def forward(self, im1, im2):
        inp = torch.cat([im1, im2], dim=1)
        return self.flowNet(inp)['flows_fw'][0]

class GMFlow(nn.Module):
    def __init__(self, model_path='checkpoints/gmflow_things.pth'):
        super(GMFlow, self).__init__()
        
        from flow_models.gmflow import gmflow
        chkpt_path = pathlib.Path(model_path)

        flowNet = gmflow.GMFlow(feature_channels=128, num_scales=1, upsample_factor=8, num_head=1,
                                 attention_type='swin', ffn_dim_expansion=4, num_transformer_layers=6)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        flowNet.load_state_dict(checkpoint['model'])
        self.flowNet = flowNet

    def forward(self, im1, im2):
        im1 = im1 * 255
        im2 = im2 * 255

        results_dict = self.flowNet(im1, im2, attn_splits_list=[2], corr_radius_list=[-1],
                                    prop_radius_list=[-1], pred_bidir_flow=False)
        return results_dict['flow_preds'][-1]

class PWC(nn.Module):
    def __init__(self):
        super(PWC, self).__init__()
        from flow_models.pwcnet.pwc import Network
        self.flowNet = Network().eval().cpu()
        
    def forward(self, im1, im2):
        im1 = im1.squeeze()
        im2 = im2.squeeze()

        intWidth, intHeight = im1.shape[2], im1.shape[1]

        tenPreprocessedFirst = im1.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = im2.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.ceil(intWidth / 64.0) * 64.0)
        intPreprocessedHeight = int(math.ceil(intHeight / 64.0) * 64.0)

        tenPreprocessedFirst = F.interpolate(tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedSecond = F.interpolate(tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tenFlow = 20.0 * F.interpolate(self.flowNet(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow

def normalize_flow(flow):
    _, _, h, w = flow.shape
    device = flow.device

    base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
    base = torch.stack(base).float().to(device)

    flow = flow + base
    size = torch.tensor([w, h]).float().to(device)
    flow = -1 + 2.*flow/(-1 + size)[:, None, None]
    return flow.permute(0, 2, 3, 1)

def warp(im, flow, padding_mode='reflection'):
    return F.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

if __name__ == '__main__':
    flowNet = ARFlow()
    im1 = torch.randn(8, 3, 512, 512)
    im2 = torch.randn(8, 3, 512, 512)
    flow = flowNet(im1, im2)
    print(flow.shape)
