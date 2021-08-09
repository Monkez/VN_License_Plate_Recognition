from tensorflow.keras.models import load_model
from modules.transform import four_point_transform
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

model = load_model("models/alignment/model.h5", compile=False)
model2 = load_model("models/alignment/model_x.h5", compile=False)
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
        
def find_corner(crop):
    h = crop.shape[0]
    w = crop.shape[1]
    x_pos = np.zeros(h)
    for i in range(h):
        for j in range(w):
            if j == w-1:
                x_pos[i]=j
            if crop[i, j]>5:
                x_pos[i]=j
                break
        
    sum_dis = np.arange(h)+x_pos
    argmin = np.argmin(sum_dis)
    corner = (int(x_pos[argmin]), argmin)
    return corner

def transform_with_up_bottom_points(mask, up_points, bottom_points):
    h = mask.shape[0]
    straighted = np.zeros_like(mask)
    for i in range(len(up_points)-1):
        _w = int(up_points[i+1][0] - up_points[i][0])
        rect = np.array([up_points[i], up_points[i+1], bottom_points[i+1], bottom_points[i]] , dtype="float32")
        dst = np.array([[0, 0], [_w, 0],  [_w, h], [0, h]] , dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_i = cv2.warpPerspective(mask, M, (_w, h))
        straighted[:, int(up_points[i][0]):int(up_points[i+1][0])] = warped_i
    return straighted

def straighten(warped, warped_origin, corners, factor=5):
    straighted = np.zeros_like(warped)
    straighted_origin = np.zeros_like(warped_origin)
    h, w = warped.shape[:2]
    H, W = warped_origin.shape[:2]
    h_r, w_r  = H/h, W/w
    up_points = [corners[0]]
    bottom_points = [corners[3]]
    UP_points = [(int(corners[0][0]*w_r), int(corners[0][1]*h_r))]
    BOTTOM_points = [(int(corners[3][0]*w_r), int(corners[3][1]*h_r))]

    for i in range(1, factor):
        up_point = None
        bottom_point = None
        for j in range(h//5):
            if warped[j, int(w/factor*i)]>10 :
                up_point = [int(w/factor*i), j]
                break

        for j in range(h//5):
            if warped[h-j-1, int(w/factor*i)]>10 :
                bottom_point = [int(w/factor*i), h-j-1]
                break

        if up_point is not None and bottom_point is not None:
            up_points.append(up_point)
            UP_points.append((int(up_point[0]*w_r), int(up_point[1]*h_r)))
            bottom_points.append(bottom_point)
            BOTTOM_points.append((int(bottom_point[0]*w_r), int(bottom_point[1]*h_r)))

    up_points.append(corners[1])
    bottom_points.append(corners[2])        
    UP_points.append((int(corners[1][0]*w_r), int(corners[1][1]*h_r)))
    BOTTOM_points.append((int(corners[2][0]*w_r), int(corners[2][1]*h_r)))

    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    for point in up_points:
        cv2.circle(warped, tuple(point), 3, (255, 0, 255), -1)
    for point in bottom_points:
        cv2.circle(warped, tuple(point), 3, (255, 0, 255), -1)

    straighted= transform_with_up_bottom_points(warped, up_points, bottom_points)
    cv2.imwrite("out/straighted.jpg", straighted)
    straighted_orin= transform_with_up_bottom_points(warped_origin, UP_points, BOTTOM_points)
    return straighted_orin


def align_image(img):
    cv2.imwrite("out/img.jpg", img)
    HEIGHT = 512
    WIDTH = 512
    resized_img = cv2.resize(img, (HEIGHT, WIDTH))
    bounding0 = model.predict(np.array([resized_img / 255.0]))[0]
    bounding0 = (255 *bounding0).astype(np.uint8)
    bounding0 = cv2.resize(bounding0, (HEIGHT, WIDTH))
    cv2.imwrite("out/bounding0.jpg", bounding0)
    bounding0 = cv2.cvtColor(bounding0 ,cv2.COLOR_GRAY2RGB)
    bounding = model2.predict(np.array([bounding0 / 255.0]))[0]
    bounding = (255 *bounding).astype(np.uint8)
    bounding = cv2.resize(bounding, (HEIGHT, WIDTH))
    cv2.imwrite("out/bounding.jpg", bounding)
    kernel = np.ones((3, 3), np.uint8)
    Thres = bounding
    Thres = cv2.GaussianBlur(Thres,(5,5),0)
    Thres2 = Thres.copy()

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



    warped_mask, warped_origin = four_point_transform_with_mask(Thres, img, box)
    cv2.imwrite("out/warped_mask.jpg", warped_mask)
    kernel = np.ones((5, 5), np.uint8)
    warped_mask = cv2.erode(warped_mask, kernel, iterations=1)
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
    cv2.circle(warped_mask_draw, left_top_corner, 3, (255, 0, 255), -1)
    cv2.circle(warped_mask_draw, right_top_corner, 3, (255, 0, 255), -1)
    cv2.circle(warped_mask_draw, left_bottom_corner, 3, (255, 0, 255), -1)
    cv2.circle(warped_mask_draw, right_bottom_corner, 3, (255, 0, 255), -1)
    h, w = warped_mask_draw.shape[:2]
    H, W = warped_origin.shape[:2]
    h_r, w_r  = H/h, W/w

    rect =  np.array([list(left_top_corner), list(right_top_corner), list(right_bottom_corner), list(left_bottom_corner)], dtype = "float32")
    dst =  np.array([[0, left_top_corner[1]], [w, right_top_corner[1]], [w, right_bottom_corner[1]], [0, left_bottom_corner[1]]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(warped_mask_draw, M, (w, h))
    cv2.imwrite("out/warped_mask_erode.jpg", warped)
    RECT = np.zeros_like(rect)
    DST = np.zeros_like(dst)

    RECT[:, 0] = rect[:, 0]*w_r
    RECT[:, 1] = rect[:, 1]*h_r

    DST[:, 0] = dst[:, 0]*w_r
    DST[:, 1] = dst[:, 1]*h_r

    M2 = cv2.getPerspectiveTransform(RECT, DST)
    warped_origin2 = cv2.warpPerspective(warped_origin, M2, (W, H))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite("out/warped.jpg", warped)
    factor = 1
    if W/H>1.2:
        factor = 7
    elif W/H>1:
        factor = 5
    elif W/H>0.8:
        factor = 3
    print(factor)    
        
    straighted_orin = straighten(warped.copy(),warped_origin2, dst, factor)
    t = int(straighted_orin.shape[0]*0.015)
    straighted_orin = straighted_orin[t:-t, t:-t]
    cv2.imwrite("out/straighted_orin.jpg", straighted_orin)
    return straighted_orin
