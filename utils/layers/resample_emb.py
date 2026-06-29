
import torch
import torch.nn.functional as F
import math

def resample_patchemb(old: torch.Tensor, new_patch_len: int):

    assert old.dim() == 2, "Input tensor must be 2D (d_model, patch_size)"
    if old.size(1) == new_patch_len:
        return old

    old = old.T
    old_shape = old.size(0)
    factor = new_patch_len/old_shape
    

    def resize(x_tensor, new_shape):
        return F.interpolate(x_tensor.unsqueeze(0), size=new_shape, mode='linear').squeeze(0)

    basis_vectors = torch.eye(old_shape, dtype=torch.float32, device=old.device)
    resize_mat = resize(basis_vectors, new_patch_len).T
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)
    
    # z_inverse = z @ resize_mat_pinv
    # z_inverse_var = z_inverse.var(dim=-1).mean(dim=1).mean()
    # z_var = z.var(dim=-1).mean(dim=1).mean()
    # z_interpolate = z_inverse @ resize_mat.T
    # z_interpolate_var = z_interpolate.var(dim=-1).mean(dim=1).mean()

    # print(z_inverse_var)
    # print(z_var)
    # print(z_interpolate_var/z_inverse_var)


    resampled_kernels = resize_mat_pinv @ old * math.sqrt(factor)

    return resampled_kernels.T



# def resample_patchemb(old, new_patch_len):
#     new_patch_size = new_patch_len
#     """Resample the weights of the patch embedding kernel to target patch size."""
#     old = old.T


#     patch_size, d_model = old.shape
#     factor = new_patch_len/patch_size
    
#     if patch_size == new_patch_size:
#         return old.T


#     old = old.permute(1, 0).unsqueeze(1)


#     resampled = F.interpolate(old, size=new_patch_size, mode='linear')/math.sqrt(factor)


#     resampled = resampled.squeeze(1).permute(1, 0)

    
#     return resampled.T
