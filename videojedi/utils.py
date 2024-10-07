##############################
### author : Ge Ya (Olga) Luo
##############################

import torch

def model_cleanup(model):
    del model
    try:
        import torch.distributed as dist
        dist.destroy_process_group()
    except:
        pass
    import gc         # garbage collect library
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def feature_aggregator(model, data_loader, num_samples, filename=None):
    import numpy as np
    all_features = None
    for i, (data, target) in enumerate(data_loader):
        # data shape = (B, T, C, H, W); data range = [0, 1]
        data = data.cuda()
        feats = model.get_feats(data)
        all_features = feats if all_features is None else np.concatenate([all_features, feats], axis=0)
        if all_features.shape[0] >= num_samples:
            all_features = all_features[:num_samples]
            break
    
    if not filename is None:
        save_features(all_features, filename)
        print(f"Saved features to {filename}")
    return all_features

def save_features(features, filename):
    import numpy as np
    with open(filename, 'wb') as f:
        np.save(f, features)