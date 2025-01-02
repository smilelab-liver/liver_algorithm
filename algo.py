#OPENSLIDE_PATH = r"C:\Users\Jimmy\anaconda3\Library\openslide-win64-20231011\bin"
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import sys
import openslide
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math
import xml.etree.cElementTree as ET
from tqdm import tqdm
import openslide
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import json
from skimage.morphology import skeletonize

check_bridging = False

###################################################################
# Utility Functions
###################################################################
def read_bbox_from_json(file_path):
    """
    讀取 Bounding-Box 的 geojson 
    """
    Json = []
    import json
    with open(file_path) as f:
        datas = json.load(f)
        for data in datas:
            box = {
                "class":data["properties"]["classification"]["name"],
                "coordinates":data["geometry"]["coordinates"][0][:4]
                  }
            Json.append(box)
    return Json
def extract_annotation(line):
    """Extract the annotation details from the given line."""
    parts = line.split('"')
    name = parts[1]
    annotation_type = parts[3]
    part_of_group = parts[5]
    
    # Map partOfGroup to meaningful names
    annotation = {
        'name': 'fibrosis' if part_of_group == '0' else 'lumen',
        'partOfGroup': 'fibrosis',
        'type': annotation_type.lower()
    }
    
    return annotation
def extract_coordinates(line):
    """Extract coordinates from the given line."""
    parts = line.split('"')
    x = float(parts[3])
    y = float(parts[5])
    return [x, y]   
def read_fibrosis_xml(xml_path):
    """
    讀取 Fibrosis XML 檔案，回傳 json
    """
    json_datas = {'annotation': []}
    
    with open(xml_path, 'r') as xml_file:
        annotation = None
        coordinates = []

        for line in xml_file:
            if 'Annotation Name' in line:
                annotation = extract_annotation(line)
                coordinates = []
            elif 'Coordinate Order' in line:
                coordinate = extract_coordinates(line)
                coordinates.append(coordinate)
            elif '/Coordinates' in line and annotation is not None:
                annotation['coordinates'] = coordinates
                json_datas['annotation'].append(annotation)
                annotation = None 
    return json_datas
def parse_json(json_datas):
    """
    將 Fibrosis jsondatas 分成 fibrosis 和 lumen (內圈)
    """
    contours = json_datas['annotation']
    fibrosis_contours=[]
    lumen_contours=[]
    for contour in contours:
        if contour['name']=='fibrosis':
            fibrosis_contours.append(contour)
        elif contour['name']=='lumen':
            lumen_contours.append(contour)
        else:
            pass
    return fibrosis_contours,lumen_contours

###################################################################
# Main Functions
###################################################################
def generate_fibrosis_mask(wsi_path,xml_path,level=3):
    """
    讀取 Fibrosis xml 檔案，回傳 mask (GrayScale)
    """
    scale_rate = 2 ** level
    classes = {'duct':0 ,'portal':1, 'central':2}
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    
    # 1. Get Image Shape & Generate Mask
    wsi_data = openslide.OpenSlide(wsi_path)
    width, height = wsi_data.level_dimensions[level]
    mask = np.zeros((height, width), np.uint8)

    # 2. Read BBox Json
    json_datas = read_fibrosis_xml(xml_path)
    fibrosis_contours,lumen_contours = parse_json(json_datas)


    for contour in fibrosis_contours:
        coordinates = contour['coordinates']
        coordinates = [[int(value / scale_rate) for value in coordinate] for coordinate in coordinates]
        coordinates = np.array([coordinates], dtype=np.int32)
        mask = cv2.fillPoly(mask, coordinates, 255)
    for contour in lumen_contours:
        coordinates = contour['coordinates']
        coordinates = [[int(value / scale_rate) for value in coordinate] for coordinate in coordinates]
        coordinates = np.array([coordinates], dtype=np.int32)
        mask = cv2.fillPoly(mask, coordinates, 0)
    return mask
def generate_bbox_mask(wsi_path,json_path,level=3):
    """
    讀取 bbox geojson，回傳 bbox mask (BGR)
    """
    scale_rate = 2 ** level
    classes = {'duct':0 ,'portal':1, 'central':2}
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    names = ['duct','portal','central']
    uuid = [0,0] # [portal,central]
    
    # 1. Get Image Shape & Generate Mask
    wsi_data = openslide.OpenSlide(wsi_path)
    width, height = wsi_data.level_dimensions[level]
    mask = np.zeros((height, width, 3), np.uint8)

    # 2. Read BBox Json
    json_datas = read_bbox_from_json(json_path)

    # 3. Draw JsonDatas on Mask
    bboxes = []
    for json_data in json_datas:
        bbox_class = json_data['class']
        class_idx = classes[bbox_class]
        coordinates = []
        coordinates.append(json_data['coordinates'][0])
        coordinates.append(json_data['coordinates'][2])
        coordinates = [[round(coordinate[0]/scale_rate),round(coordinate[1]/scale_rate)] for coordinate in coordinates]
        
        x1, y1 = coordinates[0] 
        x2, y2 = coordinates[1]
        cv2.rectangle(mask, (x1, y1), (x2, y2), colors[class_idx], -1)

        if class_idx != 0:
            id = f'{names[class_idx]}#{uuid[class_idx-1]}'
            uuid[class_idx-1] += 1
            bboxes.append((x1, y1, x2, y2, class_idx,id))
        else:
            bboxes.append((x1, y1, x2, y2, class_idx,'duct'))
    return mask,bboxes,uuid
def generate_wsi_img(wsi_path,level=3):
    wsi_data = openslide.OpenSlide(wsi_path)
    width, height = wsi_data.level_dimensions[level]
    wsi_img = wsi_data.read_region((0, 0), level, (width, height))
    wsi_img = cv2.cvtColor(np.array(wsi_img), cv2.COLOR_RGB2BGR)
    return wsi_img
def cutBridge(mask, real_mask, wsi_img, bboxes, area_dict):
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    # 所有 BBoxes 的點
    all_bbox_points = set()
    for x1, y1, x2, y2, class_idx, bbox_id in bboxes:
        bbox_points_set = {(y, x) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)}
        all_bbox_points.update(bbox_points_set)

    # 每個 BBOX 的點
    bbox_points_sets = []
    for x1, y1, x2, y2, class_idx, bbox_id in bboxes:
        bbox_points_set = {(y, x) for y in range(y1-1, y2 + 2) for x in range(x1-1, x2 + 2)}
        bbox_points_sets.append(bbox_points_set)

    # 找到與所有 BBoxes 都無交集的點
    component_points = np.argwhere(mask > 0).tolist()
    component_points = {tuple(point) for point in component_points}
    non_overlapping_points = component_points - all_bbox_points
    output_mask = np.zeros_like(mask, dtype=np.uint8)
    for y, x in non_overlapping_points:
        output_mask[y, x] = 255 

    # 尋找 Bridging 的 Component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_mask)
    bridging_components = []  # 儲存 Component 中與 BBoxes 都有交集的 Component
    for label in range(1, num_labels):
        component_mask = np.zeros_like(output_mask)
        component_mask[labels == label] = 255
        component_points = np.argwhere(component_mask > 0).tolist()
        component_points = {tuple(point) for point in component_points}

        # 檢查是否與任意 BBox 都有交集
        connected_to_bboxes = [
            any(point in bbox_points_set for point in component_points)
            for bbox_points_set in bbox_points_sets
        ]

        if sum(connected_to_bboxes) == 2:
            bridging_components.append(label)
    bridging_mask = np.zeros_like(mask, dtype=np.uint8)
    for label in bridging_components:
        bridging_mask[labels == label] = 255 

    # 切割 Bridging
    x, y, w, h = cv2.boundingRect(bridging_mask)
    upper_or_left_mask = np.zeros_like(mask, dtype=np.uint8)
    lower_or_right_mask = np.zeros_like(mask, dtype=np.uint8)
    if h >= w:  
        mid_y = y + h // 2
        upper_or_left_mask[y:mid_y, x:x + w] = bridging_mask[y:mid_y, x:x + w]
        lower_or_right_mask[mid_y:y + h, x:x + w] = bridging_mask[mid_y:y + h, x:x + w]
    else:  
        mid_x = x + w // 2
        upper_or_left_mask[y:y + h, x:mid_x] = bridging_mask[y:y + h, x:mid_x]
        lower_or_right_mask[y:y + h, mid_x:x + w] = bridging_mask[y:y + h, mid_x:x + w]

    # 分配Bridging
    upper_or_left_mask_points = np.argwhere(upper_or_left_mask > 0).tolist()
    upper_or_left_mask_points = {tuple(point) for point in upper_or_left_mask_points}
    lower_or_right_mask_points = np.argwhere(lower_or_right_mask > 0).tolist()
    lower_or_right_mask_points = {tuple(point) for point in lower_or_right_mask_points}
    for x1, y1, x2, y2, class_idx, bbox_id in bboxes:
        bbox_points_set = {(y, x) for y in range(y1-1, y2 + 2) for x in range(x1-1, x2 + 2)}
        if any(point in bbox_points_set for point in upper_or_left_mask_points):
            upper_or_left_mask = cv2.bitwise_and(upper_or_left_mask, real_mask)
            # wsi_img[upper_or_left_mask > 0] = colors[class_idx]
            wsi_img[upper_or_left_mask > 0] = [255,255,0]
            area = cv2.countNonZero(component_mask)
            if area_dict.get(bbox_id) is None:
                area_dict[bbox_id] = 0
            area_dict[bbox_id] += area
            continue
        if any(point in bbox_points_set for point in lower_or_right_mask_points):
            lower_or_right_mask = cv2.bitwise_and(lower_or_right_mask, real_mask)
            # wsi_img[lower_or_right_mask > 0] = colors[class_idx]
            wsi_img[lower_or_right_mask > 0] = [0,255,255]
            area = cv2.countNonZero(component_mask)
            if area_dict.get(bbox_id) is None:
                area_dict[bbox_id] = 0
            area_dict[bbox_id] += area
            continue

    # 分配非 Bridging
    non_bridging_mask= cv2.subtract(mask,bridging_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_bridging_mask)
    for label in range(1, num_labels):
        component_mask = np.zeros_like(output_mask)
        component_mask[labels == label] = 255
        component_points = np.argwhere(component_mask > 0).tolist()
        component_points = {tuple(point) for point in component_points}
        for x1, y1, x2, y2, class_idx, bbox_id in bboxes:
            bbox_points_set = {(y, x) for y in range(y1-1, y2 + 2) for x in range(x1-1, x2 + 2)}
            if any(point in bbox_points_set for point in component_points):
                component_mask = cv2.bitwise_and(component_mask, real_mask)
                wsi_img[component_mask>0] = colors[class_idx]
                area = cv2.countNonZero(component_mask)
                if area_dict.get(bbox_id) is None:
                    area_dict[bbox_id] = 0
                area_dict[bbox_id] += area
                break
def cutBridge_v2(mask, real_mask, wsi_img, bboxes, area_dict):
    def split_mask_by_bboxes(mask, bboxes):
        """
        根據兩個 bounding box 的位置切割 mask，並返回上下或左右的 mask。
        """
        x1, y1, x2, y2, _, _ = bboxes[0]
        x3, y3, x4, y4, _, _ = bboxes[1]

        x_arr = [x1, x2, x3, x4]
        y_arr = [y1, y2, y3, y4]
        x_arr.sort()
        y_arr.sort()

        mid_x = (x_arr[1] + x_arr[2]) // 2
        mid_y = (y_arr[1] + y_arr[2]) // 2

        x, y, w, h = cv2.boundingRect(mask)

        upper_or_left_mask = np.zeros_like(mask, dtype=np.uint8)
        lower_or_right_mask = np.zeros_like(mask, dtype=np.uint8)

        # 判斷切割方向
        if h >= w:  # 垂直方向切割
            upper_or_left_mask[y:mid_y, x:x + w] = mask[y:mid_y, x:x + w]
            lower_or_right_mask[mid_y:y + h, x:x + w] = mask[mid_y:y + h, x:x + w]
        else:  # 水平方向切割
            upper_or_left_mask[y:y + h, x:mid_x] = mask[y:y + h, x:mid_x]
            lower_or_right_mask[y:y + h, mid_x:x + w] = mask[y:y + h, mid_x:x + w]

        return upper_or_left_mask, lower_or_right_mask
    def refine_mask(mask):
        """
        保留 mask 中的最大連通元件，將較小的連通元件移動到另一個 mask。
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = stats[:, cv2.CC_STAT_AREA]

        if len(areas) <= 1:  # 沒有任何有效的連通元件
            print("Warning: No valid components in the mask.")
            refined_mask = mask.copy()
            removed_mask = np.zeros_like(mask)
            return refined_mask, removed_mask

        largest_component_idx = 1 + np.argmax(areas[1:])
        max_area = areas[largest_component_idx]

        # 初始化另一個 mask 用於保存移除的元件
        refined_mask = mask.copy()
        removed_mask = np.zeros_like(mask)
        for label in range(1, num_labels):
            if areas[label] < max_area:
                component_mask = np.zeros_like(mask)
                component_mask[labels == label] = 255
                removed_mask = cv2.add(removed_mask, component_mask)
                refined_mask = cv2.subtract(refined_mask, component_mask)
        return refined_mask, removed_mask
    def assign_bridging(mask, real_mask, bboxes, wsi_img, area_dict, color_map):
        """
        根據 bboxes 的位置分配 Bridging 區域，更新影像與面積。
        """
        mask_points = set(map(tuple, np.argwhere(mask > 0)))
        for x1, y1, x2, y2, class_idx, bbox_id in bboxes:
            bbox_points_set = {(y, x) for y in range(y1 - 1, y2 + 2) for x in range(x1 - 1, x2 + 2)}
            if mask_points & bbox_points_set:  # 是否有交集
                color = color_map.get(class_idx, [255, 255, 255]) 
                mask = cv2.bitwise_and(mask, real_mask)
                wsi_img[mask > 0] = color
                area = cv2.countNonZero(mask)
                area_dict[bbox_id] = area_dict.get(bbox_id, 0) + area
                break

    upper_or_left_mask, lower_or_right_mask = split_mask_by_bboxes(mask, bboxes)
    lower_or_right_mask, moved_to_upper = refine_mask(lower_or_right_mask)
    upper_or_left_mask = cv2.add(upper_or_left_mask, moved_to_upper)
    upper_or_left_mask, moved_to_lower = refine_mask(upper_or_left_mask)
    lower_or_right_mask = cv2.add(lower_or_right_mask, moved_to_lower)

    color_map = {
        0: [255, 255, 255],
        1: [0, 0, 255],
        2: [0, 255, 0],
    }
    assign_bridging(lower_or_right_mask, real_mask, bboxes, wsi_img, area_dict, color_map)
    if check_bridging:
        color_map = {
            0: [255, 255, 255],
            1: [0, 255, 0],
            2: [0, 0, 255],
        }
    assign_bridging(upper_or_left_mask, real_mask, bboxes, wsi_img, area_dict, color_map)
def compute_area(x1, y1, x2, y2):
    """計算邊界框的面積"""
    return max(0, x2 - x1) * max(0, y2 - y1)
def compute_iou(box1, box2):
    """計算兩個邊界框之間的 IoU"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = compute_area(inter_x1, inter_y1, inter_x2, inter_y2)

    area1 = compute_area(x1, y1, x2, y2)
    area2 = compute_area(x1_p, y1_p, x2_p, y2_p)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0
def check_boxes(bboxes):
    """
    檢查邊界框是否重疊，若重疊則保留面積大的框。
    bboxes: List of bounding boxes [(x1, y1, x2, y2, class_idx, bbox_id), ...]
    """
    bboxes = sorted(bboxes, key=lambda box: compute_area(box[0], box[1], box[2], box[3]), reverse=True)
    
    keep = []  
    removed = set()  
    for i, box1 in enumerate(bboxes):
        if i in removed:
            continue
        keep.append(box1)
        for j, box2 in enumerate(bboxes[i + 1:], start=i + 1):
            if j in removed:
                continue
            # 計算 IoU
            iou = compute_iou(box1[:4], box2[:4])
            if iou > 0: 
                removed.add(j) 
    return keep
def distinguish_duct_portal(mask,wsi_img, portal_num, area_dict):
    """
    透過將不連續的纖維作膨脹，讓他們可以綁定在一起
    
    mask 為纖維上若有膽管但沒血管的部分
    wsi_img 為視覺化圖
    """
    mask_dilated = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dilated)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 1000:
            continue

        component_mask = np.zeros_like(mask_dilated)
        component_mask[labels == label] = 255

        # 計算 bounding box
        x, y, w, h = cv2.boundingRect(component_mask)
        portal_id = f"portal#{portal_num+label}"

        component_mask = cv2.bitwise_and(component_mask, mask)
        # 視覺化
        wsi_img[component_mask > 0] = (0, 0, 255)
        cv2.rectangle(wsi_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        (text_w, text_h), baseline = cv2.getTextSize(portal_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(wsi_img, (int(x), int(y) - text_h - baseline), (int(x) + text_w, int(y)), (0, 0, 255), -1)
        cv2.putText(wsi_img, portal_id, (int(x), int(y) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # 計算面積
        area = cv2.countNonZero(component_mask)
        if area_dict.get(portal_id) is None:
            area_dict[portal_id] = 0
        area_dict[portal_id] += area

    return wsi_img
def is_component_in_bbox(component_points, bboxes):
    """
    檢查 component 的所有點是否在任意 portal or central bbox 範圍內
    """
    overlap_bbox = []
    component_points = {tuple(point) for point in component_points}
    for idx, (x1, y1, x2, y2, class_idx, bbox_id) in enumerate(bboxes):
        # 排除膽管
        if class_idx == 0:
            continue
        bbox_points_set = {(y, x) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)}
        if any(point in bbox_points_set for point in component_points):
            overlap_bbox.append(idx)
    return overlap_bbox
def is_component_in_duct_bbox(component_points, bboxes):
    """
    檢查 component 的所有點是否在任意 duct bbox 範圍內
    """
    overlap_bbox = []
    component_points = {tuple(point) for point in component_points}
    for idx, (x1, y1, x2, y2, class_idx, bbox_id) in enumerate(bboxes):
        # 排除 portal or central bbox
        if class_idx != 0:
            continue
        bbox_points_set = {(y, x) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)}
        if any(point in bbox_points_set for point in component_points):
            overlap_bbox.append(idx)
    
    return overlap_bbox
def generate_tissue_skeleton(tissue_mask):
    """
    產生穿刺的骨架 => 用來檢查是否有 Bridging
    """
    tissue_mask = (tissue_mask // 255).astype(np.uint8) 
    tissue_mask = cv2.dilate(tissue_mask, np.ones((31, 31), np.uint8), iterations=1)
    skeleton0 = skeletonize(tissue_mask,method='lee')
    skeleton = (skeleton0.astype(np.uint8) * 255)
    return skeleton
def check_bridge_when_no_vein(mask):
    """
    檢查當沒血管的情況下是否有 Bridging
    """
    global tissue_skeleton
    bridge_mask = cv2.bitwise_and(tissue_skeleton, mask)
    if cv2.countNonZero(bridge_mask) > 0:
        return True
    else:
        return False
def removeedge(mask, area_threshold=30000):
    """
    移除纖維極值的部分
    """
    mask = mask.astype(np.uint8)
    # mask_origin = mask.copy()

    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)

    # Ensure that the mask is binary (0 or 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize a clean mask
    clean_mask = np.zeros_like(mask)
    for contour in contours:
        # Calculate the contour area using cv2.contourArea
        contour_area = cv2.contourArea(contour)
        
        # Keep contours within the acceptable range
        if contour_area <= area_threshold:
            # Draw the valid contour on the clean mask
            cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # # dilate
    # kernel = np.ones((5, 5), np.uint8)
    # clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)
    # # bitwise
    # clean_mask = cv2.bitwise_and(clean_mask, mask_origin)
    return clean_mask
def draw_skeleton(wsi_img):
    global tissue_skeleton
    wsi_img[tissue_skeleton == 255] = [0, 255, 255]
    return wsi_img
def process_component(component_label, labels,fibrosis_mask ,fibrosis_dilated, bboxes, wsi_img, stats, area_dict, duct_portal_mask):
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    bbox_colors = [(255, 0, 0), (255, 255, 0), (127, 0, 255)]

    # 產生目前 Component 的 mask     
    component_mask = np.zeros_like(fibrosis_dilated)
    component_mask[labels == component_label] = 255 # 這是有 Dialated 的 mask
    component_points = np.argwhere(component_mask > 0).tolist()
    real_component_mask = np.zeros_like(fibrosis_dilated)
    real_component_mask = cv2.bitwise_and(fibrosis_mask, component_mask) # 這是一般的 mask
    area = cv2.countNonZero(real_component_mask)
    # 面積太小 == 太破碎，直接 assign 成 zone2
    if area < 200:
        with lock:
            wsi_img[real_component_mask > 0] = [255, 0, 0] # 藍色
            if area_dict.get('zone2') is None:
                area_dict['zone2'] = 0
            area_dict['zone2'] += area
        return
    
    # 確認是否有重疊血管或膽管
    overlap_bboxex = is_component_in_bbox(component_points, bboxes)
    overlap_duct_bboxes = is_component_in_duct_bbox(component_points, bboxes)

    with lock:
        tmp_boxes = []
        for overlap_bbox in overlap_bboxex:
            tmp_boxes.append(bboxes[overlap_bbox])
        # 如果 BBOX 重疊就不用特別作 bridging
        # TODO: BBOX 距離太近的情況，要合併
        tmp_boxes = check_boxes(tmp_boxes)
        # 計算面積 & 視覺化纖維
        if len(tmp_boxes) == 1:
            # 一個血管的情況
            x1, y1, x2, y2, class_idx, bbox_id = tmp_boxes[0]
            wsi_img[real_component_mask > 0] = colors[class_idx] # Portal:紅色 Central:綠色
            if area_dict.get(bbox_id) is None:
                area_dict[bbox_id] = 0
            area_dict[bbox_id] += area
        elif len(overlap_bboxex) == 2:
            # TODO: 兩個血管的情況 => Briding
            cutBridge_v2(component_mask, real_component_mask, wsi_img, tmp_boxes, area_dict)
        elif len(tmp_boxes) > 2:
            wsi_img[real_component_mask > 0] = [0, 255, 255]  # 黄色
        else:
            # 沒有血管的情況
            if len(overlap_duct_bboxes) > 0 and not check_bridge_when_no_vein(component_mask):
                # TODO: 如果有膽管要判定成 Portal => 新增一個 portal#num 給他儲存結果
                duct_portal_mask[real_component_mask > 0] = 255
            else:
                if area >= 1000 and check_bridge_when_no_vein(component_mask):
                    wsi_img[real_component_mask > 0] = [255, 255, 0] # 青色
                    if area_dict.get('birdge') is None:
                        area_dict['birdge'] = 0
                    if area_dict.get('birdge_num') is None:
                        area_dict['birdge_num'] = 0
                    area_dict['birdge'] += area
                    area_dict['birdge_num'] += 1
                else:
                    wsi_img[real_component_mask > 0] = [255, 0, 0] # 藍色
                    if area_dict.get('zone2') is None:
                        area_dict['zone2'] = 0
                    area_dict['zone2'] += area
        # 畫血管 BBox
        for overlap_bbox in overlap_bboxex:
            x1, y1, x2, y2, class_idx, bbox_id = bboxes[overlap_bbox]
            cv2.rectangle(wsi_img, (x1, y1), (x2, y2), bbox_colors[class_idx], 2)
            (text_w, text_h), baseline = cv2.getTextSize(bbox_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(wsi_img, (int(x1), int(y1) - text_h - baseline), (int(x1) + text_w, int(y1)), bbox_colors[class_idx], -1)
            cv2.putText(wsi_img, bbox_id, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # 畫膽管 BBox
        for overlap_duct_bbox in overlap_duct_bboxes:
            x1, y1, x2, y2, class_idx, bbox_id = bboxes[overlap_duct_bbox]
            cv2.rectangle(wsi_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
def main_processing(labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, stats, num_labels, area_dict, duct_portal_mask):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_component, label, labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, stats, area_dict, duct_portal_mask)
            for label in range(1, num_labels)
        ]
        for future in futures:
            future.result() 
    return wsi_img,duct_portal_mask

if __name__ == "__main__":
    global tissue_skeleton
    if len(sys.argv) > 1:
        try:
            level = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer for level.")
            sys.exit(1)
    else:
        level = 1 
    json_root_path = f'/work/u3516703/liver_score/f{level}_bbox_json'
    wsi_root_path = f'/work/u3516703/liver_score/f{level}'
    xml_root_path = f'/work/u3516703/liver_score/f{level}_liver_xml'
    tissue_root_xml_path = f'/work/u3516703/liver_score/f{level}_tissue_xml/biopsy'
    save_result_vis_path = f'/work/u3516703/liver_score/f{level}_result'

    for file in os.listdir(wsi_root_path):
        if '.mrxs' in file:
            start_time = time.time()
            wsi_path = os.path.join(wsi_root_path,file)
            json_path = os.path.join(json_root_path,file.replace('.mrxs','.geojson'))
            xml_path = os.path.join(xml_root_path,file.replace('.mrxs','.xml'))
            tissue_xml_path = os.path.join(tissue_root_xml_path,file.replace('.mrxs','.xml'))

            bbox_mask,bboxes,uuid = generate_bbox_mask(wsi_path,json_path,level=4)
            fibrosis_mask = generate_fibrosis_mask(wsi_path,xml_path,level=4)    
            tissue_mask = generate_fibrosis_mask(wsi_path,tissue_xml_path,level=4)
            # TODO : 此處穿刺的骨架可能會有問題
            tissue_mask = cv2.dilate(tissue_mask, np.ones((15, 15), np.uint8), iterations=1)
            tissue_skeleton = generate_tissue_skeleton(tissue_mask)

            # 處理掉上方切片組織
            fibrosis_mask = cv2.bitwise_and(fibrosis_mask,tissue_mask)
            fibrosis_mask = removeedge(fibrosis_mask)

            wsi_img = generate_wsi_img(wsi_path,level=4)
            duct_portal_mask = np.zeros_like(fibrosis_mask)

            fibrosis_dilated = cv2.dilate(fibrosis_mask, np.ones((3, 3), np.uint8), iterations=1)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fibrosis_dilated)

            lock = threading.Lock()

            area_dict = {}
            final_image, duct_portal_mask = main_processing(labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, stats, num_labels, area_dict, duct_portal_mask)
            final_image = distinguish_duct_portal(duct_portal_mask,final_image, uuid[0], area_dict)
            if not os.path.exists(save_result_vis_path):
                os.makedirs(save_result_vis_path)
            save_path = os.path.join(save_result_vis_path,file.replace('.mrxs','.jpg'))
            final_image = draw_skeleton(final_image)
            cv2.imwrite(save_path, final_image)
            print(area_dict)
            save_path = os.path.join(save_result_vis_path,file.replace('.mrxs','.json'))
            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(area_dict, json_file, ensure_ascii=False, indent=4)
            end_time = time.time() 
            elapsed_time = end_time - start_time
            print(f"The file {file} took {elapsed_time} seconds to complete.")
