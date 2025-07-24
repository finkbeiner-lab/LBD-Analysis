from PIL import Image
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch import topk
import numpy as np
import os
import skimage.transform
import cv2
import math
import openslide
import argparse
import pandas as pd


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def cam_to_mask(gray, patches, cam_matrix, w, h, w_s, h_s):
   mask = np.full_like(gray, 0.).astype(np.float32)
   for ind1, patch in enumerate(patches):
      x, y = patch.split('.')[0].split('_')
      x, y = int(x), int(y)
      if y <5 or x>w-5 or y>h-5:
         continue
      mask[int(y*h_s):int((y+1)*h_s), int(x*w_s):int((x+1)*w_s)].fill(cam_matrix[ind1][0])

   return mask

def main(args):
   
   file_text_all = open(args.path_file, 'r').readlines()
   for i in range(len(file_text_all)): 
      file_name, label =  file_text_all[i].split('\t')      #open(args.path_file, 'r').readlines()[0].split('\t')
      #site, file_name = file_name.split('/')
      file_path = os.path.join(args.path_patches, '{}/20.0/'.format(file_name))
      print(file_name)
      print(label)

      p = torch.load('graphcam/kfold_output/{}/prob.pt'.format(file_name)).cpu().detach().numpy()[0]
      file_path = os.path.join(args.path_patches, '{}/20.0/'.format(file_name))
      #ori = openslide.OpenSlide(os.path.join(args.path_WSI, '{}.svs').format(file_name))
      patch_info = open(os.path.join(args.path_graph, file_name, 'c_idx.txt'), 'r')

      #width, height = ori.dimensions
      """
      w, h = int(width/512), int(height/512)
      w_r, h_r = int(width/40), int(height/40)
      resized_img = ori.get_thumbnail((w_r,h_r))
      resized_img = resized_img.resize((w_r,h_r))
      w_s, h_s = float(512/40), float(512/40)
      print(w_s, h_s)
      """

      patch_info = patch_info.readlines()
      patches = []
      xmax, ymax = 0, 0
      for patch in patch_info:
         x, y = patch.strip('\n').split('\t')
         if xmax < int(x): xmax = int(x)
         if ymax < int(y): ymax = int(y)
         patches.append('{}_{}.jpeg'.format(x,y))

      #output_img = np.asarray(resized_img)[:,:,::-1].copy()
      #-----------------------------------------------------------------------------------------------------#
      # GraphCAM
      print('visulize GraphCAM')
      assign_matrix = torch.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam/kfold_output/{}/s_matrix_ori.pt'.format(file_name))
      m = nn.Softmax(dim=1)
      assign_matrix = m(assign_matrix)

      # Thresholding for better visualization
      p = np.clip(p, 0.4, 1)

      # Load graphcam for differnet class
      cam_matrix_0 = torch.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam/kfold_output/{}/cam_0.pt'.format(file_name))
      print(cam_matrix_0.shape)
      cam_matrix_0 = torch.mm(assign_matrix, cam_matrix_0.transpose(1,0))
      cam_matrix_0 = cam_matrix_0.cpu()
      
      cam_matrix_1 = torch.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam/kfold_output/{}/cam_1.pt'.format(file_name))
      cam_matrix_1 = torch.mm(assign_matrix, cam_matrix_1.transpose(1,0))
      cam_matrix_1 = cam_matrix_1.cpu()

      # Normalize the graphcam
      cam_matrix_0 = (cam_matrix_0 - cam_matrix_0.min()) / (cam_matrix_0.max() - cam_matrix_0.min())
      cam_matrix_0 = cam_matrix_0.detach().numpy()
      cam_matrix_0 = p[0] * cam_matrix_0
      cam_matrix_0 = np.clip(cam_matrix_0, 0, 1)
      
      
      cam_matrix_1 = (cam_matrix_1 - cam_matrix_1.min()) / (cam_matrix_1.max() - cam_matrix_1.min())
      cam_matrix_1 = cam_matrix_1.detach().numpy()
      cam_matrix_1 = p[1] * cam_matrix_1
      cam_matrix_1 = np.clip(cam_matrix_1, 0, 1)


      """
      output_img_copy =np.copy(output_img)

      gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
      image_transformer_attribution = (output_img_copy - output_img_copy.min()) / (output_img_copy.max() - output_img_copy.min())



      mask0 = cam_to_mask(gray, patches, cam_matrix_0, w, h, w_s, h_s)
      vis0 = show_cam_on_image(image_transformer_attribution, mask0)
      vis0 =  np.uint8(255 * vis0) 
      mask1 = cam_to_mask(gray, patches, cam_matrix_1, w, h, w_s, h_s)
      vis1 = show_cam_on_image(image_transformer_attribution, mask1)
      vis1 =  np.uint8(255 * vis1)


      ##########################################
      h, w, _ = output_img.shape
      if h > w:
         vis_merge = cv2.hconcat([output_img, vis0, vis1])
      else:
         vis_merge = cv2.vconcat([output_img, vis0, vis1])

      cv2.imwrite('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam_vis/{}_all_types_cam_all.png'.format(file_name), vis_merge)

      cv2.imwrite('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam_vis/{}_all_types_ori.png'.format(file_name), output_img)
      cv2.imwrite('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam_vis/{}_all_types_cam_pdd.png'.format(file_name), vis0)
      cv2.imwrite('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam_vis/{}_all_types_cam_dlb.png'.format(file_name), vis1)
      """

      coords_x =[]
      coords_y =[]
      for patch in patch_info:
         x, y = patch.strip('\n').split('\t')
         coords_x.append(int(x)*512*2)
         coords_y.append(int(y)*512*2)

      coords_df = pd.DataFrame({"x":coords_x, "y":coords_y})
      #rint(len(coords_df))
      #print(cam_matrix_0)
      cam_df = pd.DataFrame(cam_matrix_0, columns=["cam"])
      pd.concat([coords_df, cam_df], axis=1).to_csv('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam_vis/kfold_output/{}_0'.format(file_name)+".csv")
      cam_df = pd.DataFrame(cam_matrix_1, columns=["cam"])
      pd.concat([coords_df, cam_df], axis=1).to_csv('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/graphcam_vis/kfold_output/{}_1'.format(file_name)+".csv")
      #print(len(cam_df))
      


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='GraphCAM')
   parser.add_argument('--path_file', type=str, default='test.txt', help='txt file contains test sample')
   parser.add_argument('--path_patches', type=str, default='', help='')
   parser.add_argument('--path_WSI', type=str, default='', help='')
   parser.add_argument('--path_graph', type=str, default='', help='')
   args = parser.parse_args()
   main(args)