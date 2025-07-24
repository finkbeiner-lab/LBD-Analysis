import os
import sys
import pyvips as Vips
import numpy as np
from torchvision import datasets, models, transforms
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
from numba import jit, cuda
from PIL import Image
 
class pyvips_image:
    def __init__(self, img_path, normalizer =None):
        self.img_path = img_path
        self.vips_img = Vips.Image.new_from_file(img_path, level=0)
        self.normalizer =  normalizer
    
    def getVipsInfo(self):
        # # Get bounds-x and bounds-y offeset
        vfields = [f.split('.') for f in self.vips_img.get_fields()]
        vfields = [f for f in vfields if f[0] == 'openslide']
        vfields = dict([('.'.join(k[1:]), self.vips_img.get('.'.join(k))) for k in vfields])
        return vfields
    
    def normalize_image(self):
        if self.normalizer!=None:
            norm_vips_img = self.normalizer.transform(self.vips_img)
        else:
            print("No normalizer defined")
        return norm_vips_img
    

    def create_image_crops(self, tilesize, stride):
        vinfo = self.getVipsInfo()
        orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])
        if self.normalizer!=None:
            vips_img = self.normalize_image()
        vips_array = self.vips_img.numpy()
        vips_array = vips_array[:,:,:3]
        xy_coords = [(x_val,y_val) for x_val in range(0, orig_w-stride, stride) for y_val in range(0, orig_h-stride, stride) 
                    if y_val + tilesize < orig_h and x_val + tilesize < orig_w]
        crops = [self.vips_img.crop(z[0], z[1], tilesize, tilesize)  for z in xy_coords]
        return xy_coords, crops
    
 
    def crop2img(self, crop):
        crop_array = crop.numpy()
        crop_array = crop_array[:,:,:3]
        img = Image.fromarray(crop_array.astype('uint8'), 'RGB')
        return img 
    

        