import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import torch
import pyvips as Vips
import numpy as np
from data.Reinhard import Reinhard
import cv2
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import geojson
import os
import cv2
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
import pandas as pd
from pyvips_image import pyvips_image
from visualize_utils import visualize_utils
import sys
sys.path.insert(0, '../')
#sys.path.insert(0, './')
from data.Reinhard import Reinhard
from models.model_mrcnn import _default_mrcnn_config, build_default
from visualization.explain import ExplainPredictions
from timeit import default_timer as timer 
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
from numba import jit
import pyvips


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_list):
        super(TestDataset, self).__init__()
        self.img_list = img_list
        self.transforms = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor(),
        transforms.Normalize([-0.0601, -0.0667, -0.0616],[0.9118, 0.9479, 0.9625])
        #transforms.Normalize([0.8478, 0.8328, 0.8679], [0.0932, 0.0973, 0.0825])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def __len__(self):
        return len(self.img_list)
    
    def crop2img(self, crop):
        crop_array = crop.numpy()
        crop_array = crop_array[:,:,:3]
        img = Image.fromarray(crop_array.astype('uint8'), 'RGB')
        return img
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = self.crop2img(img)
        transformed_img = self.transforms(img)
        transformed_img = torch.unsqueeze(transformed_img, dim=0)
        return transformed_img
 

def generate_mask_from_wgm_prediction(crops,xy_coords, model_wgm, device,orig_w, orig_h, tilesize, BATCH_SIZE):
    valDataset = TestDataset(crops)
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)
    pred_labels =  test(valDataset,device, model_wgm)
    masked_image = np.zeros((orig_h,orig_w))
    for pred, z in zip(pred_labels,xy_coords):
        masked_image[z[1]:z[1]+tilesize, z[0]:z[0]+tilesize] = pred
    return masked_image

"""
def select_gray_matter_mask(pyvips_fn, masked_image, orig_w, orig_h, tilesize=1024, stride=1024,area_threshold=0.5):
    masked_image = (masked_image==1).astype(int)
    gray_img_dict = dict()
    lb_count=0
    norm_vips_img = pyvips_fn.normalize_image()
    for y_val in range(0, orig_h-stride, stride):
        for x_val in range(0, orig_w-stride, stride): 
            if y_val + tilesize < orig_h and x_val + tilesize < orig_w:
                if np.sum(masked_image[x_val:x_val+tilesize, y_val:y_val+tilesize])>=int(tilesize*tilesize*area_threshold):
                    crop = norm_vips_img.crop(x_val, y_val, tilesize, tilesize)
                    cropped_img = pyvips_fn.crop2img(crop)
                    gray_img_dict[(x_val,y_val)] = np.array(cropped_img)
    return gray_img_dict
 """   

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
    points = get_points_in_contour(x,y,downscaled_w,downscaled_h,c,1024, 1024)
    crops = [(norm_vips_img.crop(x, y, 1024, 1024),x,y) for x,y in points if (y + tilesize < orig_h) and (x + tilesize < orig_w)]
    return crops
    

def run_lb_seg(model_lbd, img_path, LBD_model_path,save_results_path, gray_img_dict,vips_img_new1):
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    final_df = pd.DataFrame()
    results_path = os.path.join(save_results_path,img_path.split("/")[-1],"results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    masks_path =  os.path.join(save_results_path, img_path.split("/")[-1],"masks")
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    detections_path = os.path.join( save_results_path, img_path.split("/")[-1],"detections")
    if not os.path.exists(detections_path):
        os.makedirs(detections_path)
        
    #for i,v in gray_img_dict.items():
    for i in gray_img_dict:
        v = i[0].numpy()
        img_name =  img_path.split("/")[-1].split(".")[0]+"_"+str(i[0])+"_"+str(i[1])
        explain= ExplainPredictions(model_lbd, model_input_path = LBD_model_path, test_input_path=[v], 
                                        detection_threshold=0.65, wandb='', save_result=False, ablation_cam=False, save_thresholds=False,
                                        results_path=results_path,masks_path=masks_path, detections_path=detections_path, img_name = img_name)
        detected_img_list, boxes_list, df, _ = explain.generate_results_v1()
        if len(final_df)==0:
            final_df = df
        else:
            final_df = pd.concat([final_df, df])
        #if len(detected_img_list)>0:
        #    tiled_vips = Vips.Image.new_from_array(detected_img_list[0])
        #    vips_img_new1= vips_img_new1.insert(tiled_vips,i[1],i[2])
    final_df.to_csv(os.path.join(save_results_path, img.split("/")[-1], img.split("/")[-1].split(".")[0]+".csv"))
    #try:
    #    vips_img_new1.tiffsave(os.path.join(save_results_path, img.split("/")[-1], img.split("/")[-1]+".tiff"), tile=False, compression='lzw', bigtiff=False, pyramid=False)
    #except:
    #    vips_img_new1.tiffsave(os.path.join(save_results_path, img.split("/")[-1], img.split("/")[-1]+".tiff"), tile=False, compression='lzw', bigtiff=True, pyramid=False)


def generate_ground_truth_mask(test_json_dir,geofile,orig_h,orig_w):
    with open(os.path.join(test_json_dir,geofile)) as f:
        gj = geojson.load(f)
    filename = geofile.replace(".geojson","")
    features = gj['features']
    binary_array= np.zeros((orig_h,orig_w))*2
    for j in features:
        coords = j["geometry"]["coordinates"]
        class_type = j["properties"]["classification"]["names"][0]
        print(class_type)
        coords = np.array(coords[0]).astype(int)
        if class_type=="grey":
            cv2.drawContours(binary_array, [coords], -1, color=(1),thickness=cv2.FILLED)
        if class_type=="White":
            cv2.drawContours(binary_array, [coords], -1, color=(0),thickness=cv2.FILLED)
        if class_type=="bg":
            cv2.drawContours(binary_array, [coords], -1, color=(2),thickness=cv2.FILLED)
    return binary_array


def wsi_seg_metric(device,masked_image,binary_array, geofile, eval_dir):
    metric = MulticlassConfusionMatrix(num_classes=3).to(device)
    conf_final=torch.tensor(np.zeros((3,3))).to(device).to(torch.int64)
    for i in range(len(masked_image)):
        preds = torch.tensor(masked_image[i]).to(device).to(torch.int64)
        target = torch.tensor(binary_array[i]).to(device).to(torch.int64)
        a1 = metric(preds,target)
        conf_final = conf_final + a1     
    conf_final_np = conf_final.cpu().numpy()
    total_predicted = np.sum(conf_final_np, axis=0) 
    diag_elements = np.diag(conf_final_np)
    precision = diag_elements/total_predicted
    total_actual = np.sum(conf_final_np, axis=1) 
    recall = diag_elements/total_actual
    f1_score = (2*precision*recall)/(recall+precision)
    iou_coeff = (diag_elements)/(total_predicted+total_actual-diag_elements)
    csv_filename_tosave = "Eval_Metric_"+ geofile.split(".")[0] + ".csv"
    eval_metrics = pd.DataFrame({"Class":["White","Grey","bg"],"Precision":precision, "recall":recall,"f1_score":f1_score,"iou_coeff":iou_coeff})
    eval_metrics.to_csv(os.path.join(eval_dir,csv_filename_tosave))
    return conf_final_np, eval_metrics  

def run_segmentation(img_path, tilesize,stride, BATCH_SIZE, model_lbd, model_wgm,device,normalizer,LBD_model_path,save_results_path, wgm_ground_truth=False):
    start = timer()
    pyvips_fn = pyvips_image(img_path, normalizer = normalizer)
    xy_coords, crops = pyvips_fn.create_image_crops(tilesize, stride)
    vinfo = pyvips_fn.getVipsInfo()
    orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height']) 
    print("image loading time",timer()-start)
    start = timer()
    masked_image = generate_mask_from_wgm_prediction(crops, xy_coords, model_wgm, device, orig_w, orig_h, tilesize, BATCH_SIZE)
    print("WGM Segmentation time",timer()-start)
    start = timer()
    path_2_save = os.path.join(save_results_path,img_path.split("/")[-1])
    if not os.path.exists(path_2_save):
        os.makedirs(path_2_save)
    vis = visualize_utils(vips_array=pyvips_fn.vips_img.numpy(), big_mask_array=masked_image)
    masked_image_3d = vis.downscaled_mask(save_path=path_2_save)
    orig_img_np = vis.downscaled_original_image(save_path=path_2_save)
    blended_img = vis.blend_mask_with_orig(masked_image_3d,orig_img_np,alpha=0.9,save_path=path_2_save)
    print("Image and Mask saving time",timer()-start)
    start = timer()
    print("Select gray matter time",timer()-start)
    start = timer()
    vips_array = pyvips_fn.vips_img.numpy()
    vips_array = vips_array[:,:,:3]
    vips_array_copy = vips_array.copy()
    vips_img_new1 = Vips.Image.new_from_array(vips_array_copy) 
    vinfo = pyvips_fn.getVipsInfo()
    orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])
    gray_img_tuple = select_gray_matter_mask(pyvips_fn, vips_array_copy, masked_image, orig_w,orig_h,tilesize=1024, stride=1024,area_threshold=0.5)
    ### Run if we have grounfd_truth
    if wgm_ground_truth:
        geofile = img_path.split("/")[-1] + ".geojson"
        binary_array = generate_ground_truth_mask(test_json_dir,geofile,orig_h,orig_w)
        print("-------- Ground truth mask complete --------")
        conf_final_np, eval_metrics  = wsi_seg_metric(device,masked_image,binary_array, geofile, path_2_save)
    run_lb_seg(model_lbd, img_path, LBD_model_path,save_results_path, gray_img_tuple,vips_img_new1)
    print("LB Segmentation time",timer()-start)
    

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


def test(dataloader,device, model):
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


#def main():
if __name__ == '__main__':
    start = timer()
    wsi_home_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/"
    REF_IMG_PATH =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/./DLB_cases/11_063_CG_aSyn_x200.svs'
    dlb_wsi_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases"
    pdd_wsi_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases"
    WM_model_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/3uctrn1o10.pth'
    test_json_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Testing_results/test_annotations"
    #WM_model_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/1m1bkrw240.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE =256
    stride = 256
    tilesize = 1024
    LBD_model_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/models/mrcnn_models/woven-wood-253_mrcnn_model_24.pth'
    save_results_path="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Testing_results/end2end_model_preds/"+LBD_model_path.split("/")[-1]+"_"+WM_model_path.split("/")[-1]+"_updated_cropping"
    
    normalizer = normalization(REF_IMG_PATH)
    print("normalization time:", timer()-start)
    model_wgm = load_saved_model(WM_model_path)
    model_wgm = model_wgm.to(device)
    test_config = dict(batch_size = 1, num_classes = 2)
    model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
    model_lbd = build_default(model_config, im_size=1024)
    #random_dlb_wsi = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Intermediate_data/Random_WSI_PDD.csv")
    random_dlb_wsi = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Intermediate_data/Random_WSI_DLB.csv")
    random_dlb_wsi_non_entcx = random_dlb_wsi[random_dlb_wsi["region"]!="EntCx"]
    #print(len(random_dlb_wsi_non_entcx))
    imgs = random_dlb_wsi_non_entcx["WSI_name"].values
    #imgs = ["/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases/DLB_cases/04_028_Syn1_CG_200x.svs"]
     #   ,    "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases/15_134_FCx_aSyn_x200.svs"]

    """ 
    ["/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/DLB_cases/DLB_cases/04_028_Syn1_CG_200x.svs",
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
    """
    #imgs = ["/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD_Dataset/LBD/PDD_cases/PDD_cases/PD258_Syn1_PCx.svs"]
    for i, img in enumerate(imgs):
        print("************************* ", i ,' ************************')
        print("image name: ", img)
        try:
            run_segmentation(img, tilesize,stride, BATCH_SIZE, model_lbd, model_wgm,device,normalizer,LBD_model_path, save_results_path, False)
        except pyvips.error.Error as err:
            print("pyvips.error.Error:",err)
            print("Could not run for:", img)
            f = open("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Results/filenotrun.txt", "a")
            f.write(img)
            f.close()
    print("Total Time:", timer()-start )
    