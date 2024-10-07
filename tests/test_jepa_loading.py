##############################
### author : Ge Ya (Olga) Luo
##############################

import pytest
import os

def test_load_and_unload_vjepa():
    from videojedi import VJEPA
    _vjepa = VJEPA(model_dir='/network/scratch/x/xuolga/Results/')

    print(_vjepa.model_dir)
    assert os.path.exists(f"{_vjepa.model_dir}/vith16.pth.tar"), f"{_vjepa.model_dir}/vith16.pth.tar does not exist"
    assert os.path.exists(f"{_vjepa.model_dir}/ssv2-probe.pth.tar"), f"{_vjepa.model_dir}/ssv2-probe.pth.tar does not exist"

    from videojedi import model_cleanup
    model_cleanup(_vjepa)
    assert os.environ.get("_vjepa") is None


def test_vjepa_inference_240x320():
    from videojedi import VJEPA
    _vjepa = VJEPA(model_dir='/network/scratch/x/xuolga/Results/')

    # Get UCF101 loader
    import torch
    import numpy as np
    def custom_collate(batch):
        import torch
        videos, labels = [], []
        for video, _, label in batch:
            videos.append(video)
            labels.append(label)
        return torch.utils.data.dataloader.default_collate(videos), torch.utils.data.dataloader.default_collate(labels)
    
    def get_UCF_loader(video_dir, annotation_dir, frames_per_clip, train=True, batch_size=20, image_size=(240, 320)):
        from torchvision.datasets import UCF101
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import av
        import torch.nn as nn

        transform = transforms.Compose([
                    # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
                    # scale in [0, 1] of type float
                    transforms.Lambda(lambda x: x / 255.),
                    # reshape into (T, C, H, W) for easier convolutions
                    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                    # rescale to the most common size
                    transforms.Lambda(lambda x: nn.functional.interpolate(x, image_size)),
        ])
        ucf_train = UCF101(video_dir, 
                        annotation_dir,
                        frames_per_clip=frames_per_clip, train=train, transform=transform)
        return DataLoader(ucf_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    data_loader = get_UCF_loader('/network/scratch/x/xuolga/Datasets/UCF-101/videos/',
                                 '/network/scratch/x/xuolga/Datasets/UCF-101/annotations/ucfTrainTestlist/',
                                 frames_per_clip=32, train=True, batch_size=10, image_size=(240, 320))
    
    # Get a batch of data
    data, target = next(iter(data_loader))
    # Run inference
    data = data.cuda()
    feats = _vjepa.get_feats(data)
    assert feats.shape == (10, 1280)
    
    from videojedi import model_cleanup
    model_cleanup(_vjepa)

def test_vjepa_inference_224x224():
    from videojedi import VJEPA
    _vjepa = VJEPA(model_dir='/network/scratch/x/xuolga/Results/')

    # Get UCF101 loader
    import torch
    import numpy as np
    def custom_collate(batch):
        import torch
        videos, labels = [], []
        for video, _, label in batch:
            videos.append(video)
            labels.append(label)
        return torch.utils.data.dataloader.default_collate(videos), torch.utils.data.dataloader.default_collate(labels)
    
    def get_UCF_loader(video_dir, annotation_dir, frames_per_clip, train=True, batch_size=20, image_size=(240, 320)):
        from torchvision.datasets import UCF101
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import av
        import torch.nn as nn

        transform = transforms.Compose([
                    # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
                    # scale in [0, 1] of type float
                    transforms.Lambda(lambda x: x / 255.),
                    # reshape into (T, C, H, W) for easier convolutions
                    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                    # rescale to the most common size
                    transforms.Lambda(lambda x: nn.functional.interpolate(x, image_size)),
        ])
        ucf_train = UCF101(video_dir, 
                        annotation_dir,
                        frames_per_clip=frames_per_clip, train=train, transform=transform)
        return DataLoader(ucf_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    data_loader = get_UCF_loader('/network/scratch/x/xuolga/Datasets/UCF-101/videos/',
                                 '/network/scratch/x/xuolga/Datasets/UCF-101/annotations/ucfTrainTestlist/',
                                 frames_per_clip=32, train=True, batch_size=10, image_size=(224, 224))
    
    # Get a batch of data
    data, target = next(iter(data_loader))
    # Run inference
    data = data.cuda()
    feats = _vjepa.get_feats(data)
    assert feats.shape == (10, 1280)
    
    from videojedi import model_cleanup
    model_cleanup(_vjepa)

def test_vjepa_feature_aggregator():
    from videojedi import VJEPA
    _vjepa = VJEPA(model_dir='/network/scratch/x/xuolga/Results/')

    # Get UCF101 loader
    import torch
    import numpy as np
    def custom_collate(batch):
        import torch
        videos, labels = [], []
        for video, _, label in batch:
            videos.append(video)
            labels.append(label)
        return torch.utils.data.dataloader.default_collate(videos), torch.utils.data.dataloader.default_collate(labels)
    
    def get_UCF_loader(video_dir, annotation_dir, frames_per_clip, train=True, batch_size=20, image_size=(240, 320)):
        from torchvision.datasets import UCF101
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import av
        import torch.nn as nn

        transform = transforms.Compose([
                    # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
                    # scale in [0, 1] of type float
                    transforms.Lambda(lambda x: x / 255.),
                    # reshape into (T, C, H, W) for easier convolutions
                    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                    # rescale to the most common size
                    transforms.Lambda(lambda x: nn.functional.interpolate(x, image_size)),
        ])
        ucf_train = UCF101(video_dir, 
                        annotation_dir,
                        frames_per_clip=frames_per_clip, train=train, transform=transform)
        return DataLoader(ucf_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    data_loader = get_UCF_loader('/network/scratch/x/xuolga/Datasets/UCF-101/videos/',
                                 '/network/scratch/x/xuolga/Datasets/UCF-101/annotations/ucfTrainTestlist/',
                                 frames_per_clip=32, train=True, batch_size=10, image_size=(240, 320))
    
    # Get 50 samples of features
    from videojedi import feature_aggregator
    feats = feature_aggregator(_vjepa, data_loader, num_samples=50, filename='train.npy')
    assert feats.shape == (50, 1280), 'Expected 50 samples of 1280 features'
    assert os.path.exists('train.npy'), 'train.npy does not exist'

    from videojedi import model_cleanup
    model_cleanup(_vjepa)