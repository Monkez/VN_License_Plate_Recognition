from tensorflow.keras.models import load_model
from modules.transform import four_point_transform
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage.morphology import skeletonize
import thinplate as tps
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

model1 = load_model("models/alignment/model.h5", compile=False)
# model2 = load_model("model2.h5", compile=False)
model3 = load_model("models/alignment/model_x.h5", compile=False)
ZERO_IMAGE =  np.empty((0,0,3))
SHOW_IMAGE = True

def four_point_transform_with_mask(mask, orin, box):
    warped_mask = four_point_transform(mask, box)
    mask_shape = mask.shape
    orin_shape = orin.shape
    w, h = mask_shape[0:2]
    W, H = orin_shape[0:2]
    rh = H / h
    rw = W / w
    BOX = np.zeros_like(box)
    BOX[:, 0] = box[:, 0] * rh
    BOX[:, 1] = box[:, 1] * rw
    BOX = BOX.astype(np.int16)
    warped_orin = four_point_transform(orin, BOX)
    return warped_mask, warped_orin

def show(np_img, show=SHOW_IMAGE):
    if show:
        Image.fromarray(np_img).show()

def thin(blobs):
    blobs = cv2.GaussianBlur(blobs,(5,5), 0)
    ret3,blobs = cv2.threshold(blobs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blobs = cv2.cvtColor(blobs, cv2.COLOR_GRAY2BGR)
    blobs = skeletonize(blobs)
    blobs = cv2.cvtColor(blobs, cv2.COLOR_BGR2GRAY)
    return blobs

def dis(A, B):
    return np.sqrt((A[0]-B[0])**2+ (A[1]-B[1])**2)

def find_corner(crop):
    h = crop.shape[0]
    w = crop.shape[1]
    x_pos = np.zeros(h)
    for i in range(h):
        for j in range(w):
            if j == w-1:
                x_pos[i]=j
            if crop[i, j]> 50:
                x_pos[i]=j
                break
        
    sum_dis = np.arange(h)+x_pos
    argmin = np.argmin(sum_dis)
    corner = (int(x_pos[argmin]), argmin)
    return corner

def align_image(img, factors= (4, 4), out_shape = None):
    t0 =time.time()
    cv2.imwrite("out/img.jpg", img)
    HEIGHT = 512
    WIDTH = 512
    resized_img = cv2.resize(img, (HEIGHT, WIDTH))
    bounding1 = model1.predict(np.array([resized_img / 255.0]))[0]
    bounding1 = (255 *bounding1).astype(np.uint8)
    bounding0 = cv2.resize(bounding1, (HEIGHT, WIDTH))
    cv2.imwrite("out/bounding0.jpg", bounding0)
    bounding0 = cv2.cvtColor(bounding0 ,cv2.COLOR_GRAY2RGB)
    bounding = model3.predict(np.array([bounding0 / 255.0]))[0]
    bounding = (255 *bounding).astype(np.uint8)
    bounding = cv2.resize(bounding, (HEIGHT, WIDTH))
    cv2.imwrite("out/bounding.jpg", bounding)
    kernel = np.ones((3, 3), np.uint8)
    Thres = thin(bounding)
    Thres2 = Thres.copy()
    t1 =time.time()

    cv2.imwrite("out/Thres.jpg", Thres)
    
    contours,hierarchy = cv2.findContours(Thres,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt  in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        im = cv2.drawContours(Thres2,[box],0,(255,255,255), -1)
    cv2.imwrite("out/Thres2.jpg", Thres2)
    contours,hierarchy = cv2.findContours(Thres2,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    areas = np.zeros(len(contours))
    for i  in range(len(contours)):
        areas[i] = cv2.contourArea(contours[i])
    sorted_area = np.sort(areas)
    biggest_cnt = contours[np.argmax(areas)]
    rect = cv2.minAreaRect(biggest_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)


    t2= time.time()
    warped_mask, warped_origin = four_point_transform_with_mask(Thres, img, box)
    cv2.imwrite("out/warped_origin.jpg", warped_origin)
    warped_mask_draw = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
    h, w = warped_mask.shape[0:2]
    pad = 4
    left_top_crop = warped_mask[:h//pad, :w//pad]

    right_top_crop = warped_mask[:h//pad, -w//pad:]
    right_top_crop_flip = cv2.flip(right_top_crop, 1)

    left_bottom_crop = warped_mask[-h//pad:, :w//pad]
    left_bottom_crop_flip = cv2.flip(left_bottom_crop, 0)

    right_bottom_crop = warped_mask[-h//pad:, -w//pad:]
    right_bottom_crop_flip = cv2.flip(right_bottom_crop, -1)

    left_top_corner = find_corner(left_top_crop)
    right_top_corner_flip = find_corner(right_top_crop_flip)
    left_bottom_corner_flip = find_corner(left_bottom_crop_flip)
    right_bottom_corner_flip = find_corner(right_bottom_crop_flip)

    right_top_corner = (w-right_top_corner_flip[0], right_top_corner_flip[1])
    left_bottom_corner = (left_bottom_corner_flip[0],h- left_bottom_corner_flip[1])
    right_bottom_corner = (w- right_bottom_corner_flip[0],h- right_bottom_corner_flip[1])
    v_factor = factors[0]
    h_factor = factors[1]
    
    up_points = find_set_point_up(left_top_corner[0], right_top_corner[0], h_factor, warped_mask)
    up_points = [left_top_corner]+up_points+[right_top_corner]

    
    
    down_points = find_set_point_up(left_bottom_corner[0], right_bottom_corner[0], h_factor, cv2.flip(warped_mask, 0))
    down_points = [(point[0], warped_mask.shape[0]- point[1]) for point in down_points]
    down_points = [left_bottom_corner]+down_points+[right_bottom_corner]

        
    right_points = find_set_point_up(right_top_corner[1], right_bottom_corner[1], v_factor,  cv2.rotate(warped_mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
    right_points = [(warped_mask.shape[1] - point[1], point[0]) for point in right_points] 
    right_points = [right_top_corner]+right_points+[right_bottom_corner]

        
    left_points = find_set_point_up(warped_mask.shape[0] - left_bottom_corner[1], warped_mask.shape[0] - left_top_corner[1], v_factor,  cv2.rotate(warped_mask, cv2.ROTATE_90_CLOCKWISE))
    left_points = [(point[1],warped_mask.shape[0]-  point[0]) for point in left_points]   
    left_points = [left_bottom_corner]+left_points+[left_top_corner]
 
        
    H, W = warped_mask.shape[:2]
    up_distances = [dis(up_points[i], up_points[i+1]) for i in range(len(up_points)-1)]
    up_dis_sum = sum(up_distances)
    up_distances_acc = []
    s = 0
    for i in range(len(up_distances)):
        s = s+up_distances[i]
        up_distances_acc.append(s)
    up_target_points = [(int(up_distances_acc[i]/up_dis_sum*W), 0) for i in range(len(up_distances))]
    

        
    right_distances = [dis(right_points[i], right_points[i+1]) for i in range(len(right_points)-1)]
    right_dis_sum = sum(right_distances)
    right_distances_acc = []
    s = 0
    for i in range(len(right_distances)):
        s = s+right_distances[i]
        right_distances_acc.append(s)
    right_target_points = [(W, int(right_distances_acc[i]/right_dis_sum*H)) for i in range(len(right_distances))]
    

        
        
    down_distances = [dis(down_points[i], down_points[i+1]) for i in range(len(down_points)-1)]
    down_dis_sum = sum(down_distances)
    down_distances_acc = []
    s = 0
    for i in range(len(down_distances)):
        s = s+down_distances[i]
        down_distances_acc.append(s)
    down_target_points = [(int(down_distances_acc[i]/down_dis_sum*W), H) for i in range(len(down_distances))]
        
    left_distances = [dis(left_points[i], left_points[i+1]) for i in range(len(left_points)-1)]
    left_dis_sum = sum(left_distances)
    left_distances_acc = []
    s = 0
    for i in range(len(left_distances)):
        s = s+left_distances[i]
        left_distances_acc.append(s)
    left_target_points = [(0, H- int(left_distances_acc[i]/left_dis_sum*H)) for i in range(len(left_distances))]
    
    
        
    
    
    src_points = up_points[1:]+right_points[1:]+ down_points[:-1]+left_points[1:]
    dst_points = up_target_points+ right_target_points+ [(0, H)]+down_target_points[:-1]+left_target_points
    
    
    for point in src_points:
        cv2.circle(warped_mask_draw, point, 3, (0, 0, 200), -1)
        
    for point in dst_points:
        cv2.circle(warped_mask_draw, point, 3, (255, 0, 0), -1)
        
    for i in range(len(src_points)):
        cv2.line(warped_mask_draw, src_points[i], dst_points[i], (0, 255, 0), 1)
        
    cv2.imwrite("out/warped_mask_draw.jpg", warped_mask_draw)
    
    
    
    src_points = np.array([np.array([x[0]/W, x[1]/H]) for x in src_points])
    dst_points = np.array([np.array([x[0]/W, x[1]/H]) for x in dst_points])
    

    if out_shape is  not None:
        warped_origin = cv2.resize(warped_origin, out_shape)
    else:
        max_size = 1500
        h, w = warped_origin.shape[:2] 
        if max(h, w)>max_size and h>w:
            warped_origin = cv2.resize(warped_origin, (int(w/h*max_size), max_size))
        elif max(h, w)>max_size and w>h:
            warped_origin = cv2.resize(warped_origin, (max_size, int(h/w*max_size)))
    
    straighted_orin = warp_image_cv(warped_origin, src_points, dst_points)
    t3 = time.time()
    
    t = int(straighted_orin.shape[0]*0.01)
    if t<1:
        t = 1
    straighted_orin = straighted_orin[t:-t, t:-t]
    cv2.imwrite("out/straighted_orin.jpg", straighted_orin)
    t4 = time.time()
    print(t1-t0, t2-t1, t3-t2, t4-t3, t4-t0)
    return straighted_orin

def find_set_point_up(left, right, factor, img):
    points = []
    thes = img.shape[0]//4
    step = (right-left)//(factor+1)
    for i in range(factor):
        x = left+ (i+1)*step
        for j in range(0, thes):
            if img[j, x]>50:
                points.append((x, j))
                break
    return points

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
