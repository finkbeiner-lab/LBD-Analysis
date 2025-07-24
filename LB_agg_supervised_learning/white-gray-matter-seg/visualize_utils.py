import os
import numpy as np
from PIL import Image
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
from numba import jit

class visualize_utils:
    def __init__(self, vips_array=None, big_mask_array=None, color_list=None):
        self.image_array = vips_array
        self.color_list = color_list
        self.big_mask_array = big_mask_array
        if self.color_list==None:
            self.color_list=([128,0,0],[0,128,0],[244, 187, 68])
    
    @jit
    def create_colored_mask(self, array_mask_2d):
        masked_image_3d = np.zeros((array_mask_2d.shape[0],array_mask_2d.shape[1], 3))
        object_ids=list(np.unique(array_mask_2d))
        for object_id in object_ids:
            binary_mask = array_mask_2d == object_id
            masked_image_3d[binary_mask] = self.color_list[object_id]
        return masked_image_3d
    
    @jit
    def downscale_image(self, image_array):
        masked_image_tn = Image.fromarray(image_array.astype(np.uint8))
        masked_image_tn.thumbnail((1000,1000))
        masked_image_tn_np = np.array(masked_image_tn)
        return masked_image_tn_np
    
    @jit
    def downscaled_mask(self, save_path=None):
        masked_image_tn_np = self.downscale_image(self.big_mask_array)
        masked_image_3d = self.create_colored_mask(masked_image_tn_np)
        if save_path:
            masked_image_3d1 = Image.fromarray(masked_image_3d.astype(np.uint8))
            masked_image_3d1.save(os.path.join(save_path,"mask.jpg"))
        return masked_image_3d 
    
    @jit
    def downscaled_original_image(self, save_path=None):
        downscaled_img_array = self.downscale_image(self.image_array)
        if save_path:
            downscaled_img = Image.fromarray(downscaled_img_array.astype(np.uint8)).convert('RGB')
            downscaled_img.save(os.path.join(save_path,"original.jpg"))
        return downscaled_img_array[:,:, :3]

    @jit
    def blend_mask_with_orig(self, masked_image_3d, orig_img_np, alpha=0.9, save_path=None):
        blended =   (1 - alpha) * masked_image_3d + alpha * orig_img_np
        blended_img = Image.fromarray((blended).astype("uint8"))
        if save_path:
            blended_img.save(os.path.join(save_path,"overlayed.jpg"))
        return blended_img
    
    @jit
    def create_gray_matter_mask(self, masked_image):
        masked_image_grey = np.where(masked_image==1,255,0)
        masked_image_gry = Image.fromarray(masked_image_grey.astype(np.uint8))
        masked_image_gry.thumbnail((1000,1000))
        return masked_image_gry
    
    