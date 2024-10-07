##############################
### author : Ge Ya (Olga) Luo
##############################

import pytest
import os

def test_compute_metric():
    from videojedi import JEDiMetric
    import numpy as np
    jedi = JEDiMetric()
    jedi.train_features = np.random.rand(5000, 512)
    jedi.test_features = np.random.rand(5000, 512)
    assert jedi.compute_metric() >= 0
    print(f"JEDi Metric: {jedi.compute_metric()}")

def test_load_train_and_test_features():
    from videojedi import JEDiMetric
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
    
    train_loader = get_UCF_loader('/network/scratch/x/xuolga/Datasets/UCF-101/videos/',
                                  '/network/scratch/x/xuolga/Datasets/UCF-101/annotations/ucfTrainTestlist/',
                                  frames_per_clip=32, train=True, batch_size=10, image_size=(240, 320))
    test_loader = get_UCF_loader('/network/scratch/x/xuolga/Datasets/UCF-101/videos/',
                                 '/network/scratch/x/xuolga/Datasets/UCF-101/annotations/ucfTrainTestlist/',
                                 frames_per_clip=32, train=False, batch_size=10, image_size=(240, 320))
    
    jedi = JEDiMetric(feature_path='.', model_dir='/network/scratch/x/xuolga/Results/')
    jedi.load_features(train_loader=train_loader, test_loader=test_loader, num_samples=50)

    assert hasattr(jedi, 'train_features'), "train_features is not loaded"
    assert hasattr(jedi, 'test_features'), "test_features is not loaded"
    assert jedi.train_features.shape == (50, 1280)
    assert jedi.test_features.shape == (50, 1280)

    assert jedi.compute_metric() >= 0
    print(f"JEDi Metric: {jedi.compute_metric()}")