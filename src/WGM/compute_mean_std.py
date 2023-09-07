import torch
import tqdm


def compute_mean_std(train_loader, dataset_size,image_size):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    # loop through images
    for inputs in tqdm(train_loader):
        s = inputs[0].squeeze(0)
        psum    += s.sum(axis        = [0, 2, 3])
        psum_sq += (s ** 2).sum(axis = [0, 2, 3])
        
    count =  dataset_size* image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
    return total_mean, total_std


#compute_mean_std(train_loader, dataset_sizes["train"], 1024)