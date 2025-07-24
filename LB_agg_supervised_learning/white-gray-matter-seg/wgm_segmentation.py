import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import torch
import pyvips as Vips
import numpy as np
import cv2
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
from numba import jit
import pyvips
import pandas as pd
from pyvips_image import pyvips_image
from TestDataset import TestDataset
from visualize_utils import visualize_utils
from timeit import default_timer as timer 
from data.Reinhard import Reinhard
from PIL import Image
from concurrent.futures import ThreadPoolExecutor



@jit
def normalization(REF_IMG_PATH):
    print("Init Normalization")
    ref_image = Vips.Image.new_from_file(REF_IMG_PATH)
    normalizer = Reinhard()
    normalizer.fit(ref_image)
    return normalizer

def load_saved_model(path):
    checkpoint = torch.load(path)
    model_ft = checkpoint["model"]
    model_ft.load_state_dict(checkpoint['state'])
    return model_ft

def get_prediction(dataloader,device, model):
    model.eval()
    pred_labels = []
    with torch.no_grad():
        for X in dataloader:
            X= X.to(device).float()
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            pred_labels.extend(preds.tolist())
            #print(preds.tolist())
    return pred_labels

def get_points_in_contour(x,y,downscaled_w,downscaled_h,cnt,stride, tilesize):
    points = []
    stride = stride
    for x_val in range(x, x+downscaled_w-stride, stride): 
        for y_val in range(y, y + downscaled_h-stride, stride): 
            inside_1 = cv2.pointPolygonTest(cnt, (x_val, y_val), False)
            inside_2 = cv2.pointPolygonTest(cnt, (x_val+tilesize, y_val+tilesize), False)
            inside_3 = cv2.pointPolygonTest(cnt, (x_val, y_val+tilesize), False)
            inside_4 = cv2.pointPolygonTest(cnt, (x_val+tilesize, y_val), False)
            if (inside_1>= 0) and (inside_2>=0) and (inside_3>=0) and (inside_4>=0) : 
                points.append((x_val, y_val)) # points time scale factor print(f'Collected {len(self.points)} points') return self.points 
    return points


def select_gray_matter_mask(pyvips_fn, vips_array_copy, masked_image, orig_w, orig_h, tilesize=1024, stride=1024,area_threshold=0.5):
    grey_image = (masked_image==1).astype(int)
    gray_img_dict = dict()
    lb_count=0
    norm_vips_img = pyvips_fn.normalize_image()
    grey_image = np.array(grey_image,dtype=np.uint8)
    contours =  cv2.findContours(grey_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print("count of contours: ", len(contours))
    c = max(contours[0], key = cv2.contourArea)
    x,y,downscaled_w,downscaled_h = cv2.boundingRect(c)
    vips_array_copy = cv2.drawContours(vips_array_copy, [c],-1, (0,0,255),40)
    vips_array_copy = cv2.rectangle(vips_array_copy,(x, y),(x + downscaled_w,y + downscaled_h),(0,255,0),30)
    points = get_points_in_contour(x,y,downscaled_w,downscaled_h,c,tilesize, tilesize)
    crops = [(norm_vips_img.crop(x, y, tilesize, tilesize),x,y) for x,y in points if (y + tilesize < orig_h) and (x + tilesize < orig_w)]
    return crops


def generate_mask_from_wgm_prediction(crops,xy_coords, model_wgm, device,orig_w, orig_h, tilesize, BATCH_SIZE):
    valDataset = TestDataset(crops)
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)
    pred_labels =  get_prediction(valDataset,device, model_wgm)
    masked_image = np.zeros((orig_h,orig_w))
    for pred, z in zip(pred_labels,xy_coords):
        masked_image[z[1]:z[1]+tilesize, z[0]:z[0]+tilesize] = pred
    return masked_image


def get_wgm_prediction(crops,xy_coords, model_wgm, device,orig_w, orig_h, tilesize, BATCH_SIZE):
    valDataset = TestDataset(crops)
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)
    pred_labels =  get_prediction(valDataset,device, model_wgm)
    print(len(crops))
    print(len(pred_labels))
    #print(pred_labels)
    crops_to_save = np.argwhere(np.array(pred_labels)==1)
    print(crops_to_save)
    return crops_to_save
    #return 



def save_crop(crop_index, img_path, crops, xy_coords, save_path):
    v = crops[crop_index].numpy()
    tile_x, tile_y = xy_coords[crop_index]

    # Proper zero-padding
    x = str(tile_x).zfill(5)  # Ensures 5-digit padding
    y = str(tile_y).zfill(5)

    img_name = f"{x}x_{y}y.png"

    im = Image.fromarray(v)
    im.save(os.path.join(save_path, img_name))  # Removed invalid mode argument
    return img_path, img_path+"."+img_name, img_path+"/"+img_name, None, tile_x, tile_y 



def save_gray_matter_crops_gigapath_format(img_path, crops_to_save, xy_coords, crops, save_path):
    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Ensure `y` is a list
        output = list(executor.map(lambda i: save_crop(i[0], img_path, crops, xy_coords, save_path), crops_to_save))
    
    return output  # Return the collected output instead of an iterator



def save_gray_matter_crops(img_path, crops_to_save,xy_coords, crops,save_path):
    for i in crops_to_save:
        v = crops[i[0]].numpy()
        x,y  = xy_coords[i[0]][0],xy_coords[i[0]][1]
        img_name =  img_path.split("/")[-1].split(".")[0]+"_"+str(x)+"_"+str(y)
        im = Image.fromarray(v)
        im.save(os.path.join(save_path,img_name+".png"), mode='RGBA')


def run_wgm_segmentation(img_path, tilesize,stride, BATCH_SIZE, model_wgm,device,normalizer,save_results_path):
    start = timer()
    pyvips_fn = pyvips_image(img_path,normalizer)
    xy_coords, crops = pyvips_fn.create_image_crops(tilesize, stride)
    print(crops[0])
    vinfo = pyvips_fn.getVipsInfo()
    orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height']) 
    print("image loading time",timer()-start)
    start = timer()
    #masked_image = generate_mask_from_wgm_prediction(crops, xy_coords, model_wgm, device, orig_w, orig_h, tilesize, BATCH_SIZE)
    print("WGM Segmentation time",timer()-start)
    #start = timer()
    path_2_save = os.path.join(save_results_path,img_path.split("/")[-1])
    if not os.path.exists(path_2_save):
        os.makedirs(path_2_save)
    #vis = visualize_utils(vips_array=pyvips_fn.vips_img.numpy(), big_mask_array=masked_image)
    #masked_image_3d = vis.downscaled_mask(save_path=path_2_save)
    #orig_img_np = vis.downscaled_original_image(save_path=path_2_save)
    #blended_img = vis.blend_mask_with_orig(masked_image_3d,orig_img_np,alpha=0.9,save_path=path_2_save)
    #print("Image and Mask saving time",timer()-start)
    #start = timer()
    #print("Select gray matter time",timer()-start)
    start = timer()
    #vips_array = pyvips_fn.vips_img.numpy()
    #vips_array = vips_array[:,:,:3]
    #vips_array_copy = vips_array.copy()
    #vips_img_new1 = Vips.Image.new_from_array(vips_array_copy) 
    #vinfo = pyvips_fn.getVipsInfo()
    #orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])
    #gray_img_tuple = select_gray_matter_mask(pyvips_fn, vips_array, masked_image, orig_w,orig_h,tilesize=1024, stride=1024,area_threshold=0.5)
    crops_to_save = get_wgm_prediction(crops,xy_coords, model_wgm, device,orig_w, orig_h, tilesize, BATCH_SIZE)
    #save_gray_matter_crops(img_path, crops_to_save, xy_coords, crops,path_2_save)
    #save_gray_matter_crops(img_path, gray_img_tuple)
    res = save_gray_matter_crops_gigapath_format(img_path, crops_to_save, xy_coords, crops,path_2_save)
    print(res)
    pd.DataFrame(res, columns=["slide_id",	"tile_id","image",	"label", "tile_x","tile_y" ]).to_csv(os.path.join(path_2_save, "dataset.csv"))




if __name__ == '__main__':
    wsi_home_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data"
    WM_model_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/3uctrn1o10.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE =16
    stride = 256
    tilesize = 256
    #save_results_path="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/wgm_crops_256x256"
    save_results_path="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/gigapath_syn1_crops"
    REF_IMG_PATH =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./DLB_cases/11_063_CG_aSyn_x200.svs'
    csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/csv_files/antibodies_data_mapping.csv"
    model_wgm = load_saved_model(WM_model_path)
    model_wgm = model_wgm.to(device)
    
    normalizer = normalization(REF_IMG_PATH)
    imgs = pd.read_csv(csv_path)["WSI_path"].values
    imgs = ['/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./DLB_cases/11_063_CG_aSyn_x200.svs']
    
    imgs = ["/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases/DLB_cases/04_028_Syn1_CG_200x.svs",
            "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases/15_134_FCx_aSyn_x200.svs",
            "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases/16_044_PCx_aSyn_x200.svs",
            "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases/PD258_Syn1_PCx.svs",
            "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases/PD268_Syn1_CG.svs",
            "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases/PD295_Syn1_FCx.svs",
            '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./DLB_cases/14_133_FCx_aSyn_x200.svs',
            '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./DLB_cases/19_053_TCx_aSyn_x200.svs',
            '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases/PD125_Syn1_PCx.svs',
            '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./PDD_cases/PDD_cases/PD110_Syn1_TCx.svs',
            ]
    
    for i, img in enumerate(imgs):
        print("************************* ", i ,' ************************')
        print("image name: ", img)
        try:
            run_wgm_segmentation(img, tilesize,stride, BATCH_SIZE, model_wgm,device,normalizer,save_results_path)
        except pyvips.error.Error as err:
            print("pyvips.error.Error:",err)
            print("Could not run for:", img)
            f = open("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/dataset/filenotrun.txt", "a")
            f.write(img)
            f.close()
            
