import sys
#TODO change hardcoding
sys.path.append('../../../')
from concurrent.futures import process
import os
import glob
from posixpath import dirname
import re
from turtle import pd
from cv2 import THRESH_BINARY_INV, THRESH_OTSU
import numpy as np
from requests import delete
import cv2
import pyvips as Vips
from tqdm import tqdm
import pyfiglet
import argparse
import pdb
import skimage.io as io
#from src.utils import vips_utils, normalize
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

import time
from timeit import default_timer as timer
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import subprocess
import geojson
import os
from PIL import Image
from Reinhard import Reinhard


   

def normalization(REF_IMG_PATH):
    print("Init Normalization")
    ref_image = Vips.Image.new_from_file(REF_IMG_PATH)
    normalizer = Reinhard()
    normalizer.fit(ref_image)
    return normalizer

def getVipsInfo(vips_img):
    # # Get bounds-x and bounds-y offeset
    vfields = [f.split('.') for f in vips_img.get_fields()]
    vfields = [f for f in vfields if f[0] == 'openslide']
    vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])
    return vfields

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

def crop_process(i, x, y, vips_orig_img, savesubdir, orig_w, orig_h):   
    print("Crop slide thread Started.", i) 
    savecroppath = os.path.join(savesubdir, f'{filename}_x_{x}_y_{y}.png')
    # row is y, col is x
    if y + tilesize < orig_h and x + tilesize < orig_w:
        # TODO change to vips cropping
        print("---------copying image---------")
        crop = vips_orig_img.crop(x, y, 1024, 1024)
        crop.write_to_file(savecroppath)
    print("Thread Stopped ", i)


def tiling(cnt,vips_img, vips_array_copy, filename, class_type, orig_w, orig_h):
    cnt = cnt.astype(np.int)
    x,y,downscaled_w,downscaled_h = cv2.boundingRect(cnt)
    vips_array_copy = cv2.drawContours(vips_array_copy, [cnt],-1, (0,0,255),40)
    vips_array_copy = cv2.rectangle(vips_array_copy,(x, y),(x + downscaled_w,y + downscaled_h),(0,255,0),30)
    points = get_points_in_contour(x,y,downscaled_w,downscaled_h,cnt,stride, tilesize)
    for x, y in points:
        #vips_array_copy = cv2.rectangle(vips_array_copy,(x, y),(x + tilesize,y + tilesize),(255,0,0),30)
        vips_array_copy = cv2.rectangle(vips_array_copy,(x, y),(x + tilesize,y + tilesize),color_codes[class_type],30)
    th = Image.fromarray(vips_array_copy)
    th.thumbnail((1000,1000))
    th.save(os.path.join(save_dir,filename.split(".")[0]+".png"))
    savesubdir = os.path.join(save_dir, class_type)
    exe = ThreadPoolExecutor(max_workers=workers)
    futures = [exe.submit(crop_process, i, x_1, y_1, vips_img, savesubdir, orig_w, orig_h) for i, (x_1, y_1) in enumerate(points)]
    done, not_done = wait(futures, return_when=ALL_COMPLETED)
    exe.shutdown()
    return "done tiling"


def tiling(cnt,vips_img, vips_array_copy, filename, class_type, orig_w, orig_h):
    cnt = cnt.astype(np.int)
    x,y,downscaled_w,downscaled_h = cv2.boundingRect(cnt)
    vips_array_copy = cv2.drawContours(vips_array_copy, [cnt],-1, (0,0,255),3)
    vips_array_copy = cv2.rectangle(vips_array_copy,(x, y),(x + downscaled_w,y + downscaled_h),(0,255,0),20)
    points = get_points_in_contour(x,y,downscaled_w,downscaled_h,cnt,stride)
    for x, y in points:
        vips_array_copy = cv2.rectangle(vips_array_copy,(x, y),(x + tilesize,y + tilesize),(255,0,0),20)
    th = Image.fromarray(vips_array_copy)
    th.thumbnail((1000,1000))
    th.save(os.path.join(save_dir,filename.split(".")[0]+".png"))
    savesubdir = os.path.join(save_dir, class_type)
    exe = ThreadPoolExecutor(max_workers=workers)
    futures = [exe.submit(crop_process, i, x_1, y_1, vips_img, savesubdir, orig_w, orig_h) for i, (x_1, y_1) in enumerate(points)]
    done, not_done = wait(futures, return_when=ALL_COMPLETED)
    exe.shutdown()
    return "done tiling"


if __name__=="main":
    pdd_wsi_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases"
    dlb_wsi_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases"
    json_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/GeoJsons"
    save_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_images3"
    imagenames = sorted(glob.glob(os.path.join(dlb_wsi_dir, './*.svs')))
    f_list = os.listdir(json_dir)
    f_list.remove('.DS_Store')
    tilesize = 1024
    workers = 10
    stride = 256
    color_codes =  {"White":(255,0,0), "grey":(255, 170, 51),"bg":(128,0,128)}


    REF_IMG_PATH = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./DLB_cases/11_063_CG_aSyn_x200.svs'
    normalizer = normalization(REF_IMG_PATH)

    for img in imagenames:
       filename = img.split("/")[-1]
        geo_name = filename+".geojson"
        if geo_name in f_list:
            with open(os.path.join(json_dir,geo_name)) as f:
                gj = geojson.load(f)
            features = gj['features']
            print(geo_name, len(features))
            if len(features)!=0:
                vips_img = Vips.Image.new_from_file(img, level=0)
                vips_img = normalizer.transform(vips_img)
                vinfo = getVipsInfo(vips_img)
                orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])
                vips_array = np.ndarray(buffer=vips_img.write_to_memory(), dtype=np.uint8, shape=(vips_img.height, vips_img.width, vips_img.bands))
                vips_array = vips_array[:,:,:3]
                vips_array_copy =vips_array.copy()
                for j in features:
                    coords = j["geometry"]["coordinates"]
                    class_type = j["properties"]["classification"]["names"][0]
                    print(len(coords[0]))
                    cnt = np.array(coords[0])
                    print(len(cnt.shape))
                    if len(cnt.shape)==2:
                        tiling(cnt,vips_img, vips_array_copy, filename, class_type, orig_w, orig_h)
                    else:
                        print(filename, cnt.shape)