OPENSLIDE_PATH = r"C:\Users\Jimmy\anaconda3\Library\openslide-win64-20231011\bin"
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import cv2
import matplotlib
matplotlib.use('TkAgg')
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
def process_component(component_label, labels,fibrosis_mask ,fibrosis_dilated, bboxes, wsi_img, stats, area_dict):
    area = stats[component_label, cv2.CC_STAT_AREA]
    
    if area < 100:
        return

    component_mask = np.zeros_like(fibrosis_dilated)
    component_mask[labels == component_label] = 255
    component_points = np.argwhere(component_mask > 0).tolist()
    overlap_bboxex = is_component_in_bbox(component_points, bboxes)
    overlap_duct_bboxes = is_component_in_duct_bbox(component_points, bboxes)
    real_component_mask = np.zeros_like(fibrosis_dilated)
    cv2.bitwise_and(fibrosis_mask, component_mask, real_component_mask)
    real_component_points = np.argwhere(real_component_mask > 0).tolist()
    area = cv2.countNonZero(real_component_mask)


    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    bbox_colors = [(255, 0, 0), (255, 255, 0), (127, 0, 255)]
    names = ['duct', 'portal', 'central']

    with lock:
        for point in real_component_points:
            x, y = point
            # TODO: 兩個血管的情況 => Briding
            if len(overlap_bboxex) > 1:
                wsi_img[x, y] = [0, 255, 255]  # 黄色
            # 一個血管的情況
            elif len(overlap_bboxex) == 1:
                x1, y1, x2, y2, class_idx, bbox_id = bboxes[overlap_bboxex[0]]
                wsi_img[x, y] = colors[class_idx]
                if area_dict.get(bbox_id) is None:
                    area_dict[bbox_id] = 0
                area_dict[bbox_id] += area
            # 沒有血管的情況
            elif len(overlap_bboxex) == 0:
                # TODO: 如果有膽管要判定成 Portal => 新增一個 portal#num 給他儲存結果
                if len(overlap_duct_bboxes) > 0:
                    wsi_img[x, y] = [0, 255, 0]
                else:
                    wsi_img[x, y] = [255, 0, 0]  

        # 畫血管 BBox
        for overlap_bbox in overlap_bboxex:
            x1, y1, x2, y2, class_idx, bbox_id = bboxes[overlap_bbox]
            label = names[class_idx]
            cv2.rectangle(wsi_img, (x1, y1), (x2, y2), bbox_colors[class_idx], 2)
            (text_w, text_h), baseline = cv2.getTextSize(bbox_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(wsi_img, (int(x1), int(y1) - text_h - baseline), (int(x1) + text_w, int(y1)), bbox_colors[class_idx], -1)
            cv2.putText(wsi_img, bbox_id, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # 畫膽管 BBox
        for overlap_duct_bbox in overlap_duct_bboxes:
            x1, y1, x2, y2, class_idx, bbox_id = bboxes[overlap_duct_bbox]
            cv2.rectangle(wsi_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
def main_processing(labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, stats, num_labels, area_dict):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_component, label, labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, stats, area_dict)
            for label in range(1, num_labels)
        ]
        for future in futures:
            future.result() 
    return wsi_img

if __name__ == "__main__":
    level = 0
    json_root_path = f'liver_score/f{level}_bbox_json'
    wsi_root_path = f'liver_score/f{level}'
    xml_root_path = f'liver_score/f{level}_liver_xml'
    tissue_root_xml_path = f'liver_score/f{level}_tissue_xml/biopsy'
    save_result_vis_path = f'liver_score/f{level}_result'

    for file in os.listdir(wsi_root_path):
        if '15-00176-Masson.mrxs' in file:
            wsi_path = os.path.join(wsi_root_path,file)
            json_path = os.path.join(json_root_path,file.replace('.mrxs','.geojson'))
            xml_path = os.path.join(xml_root_path,file.replace('.mrxs','.xml'))
            tissue_xml_path = os.path.join(tissue_root_xml_path,file.replace('.mrxs','.xml'))

            bbox_mask,bboxes,uuid = generate_bbox_mask(wsi_path,json_path,level=4)
            fibrosis_mask = generate_fibrosis_mask(wsi_path,xml_path,level=4)
            tissue_mask = generate_fibrosis_mask(wsi_path,tissue_xml_path,level=4)

            # 處理掉上方切片組織
            fibrosis_mask = cv2.bitwise_and(fibrosis_mask,tissue_mask)

            wsi_img = generate_wsi_img(wsi_path,level=4)

            portal_bbox_mask = bbox_mask[:,:,1]
            central_bbox_mask = bbox_mask[:,:,2]

            fibrosis_dilated = cv2.dilate(fibrosis_mask, np.ones((3, 3), np.uint8), iterations=1)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fibrosis_dilated)


            start_time = time.time()
            lock = threading.Lock()

            area_dict = {}
            final_image = main_processing(labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, stats, num_labels, area_dict)
            if not os.path.exists(save_result_vis_path):
                os.makedirs(save_result_vis_path)
            save_path = os.path.join(save_result_vis_path,file.replace('.mrxs','.jpg'))
            cv2.imwrite(save_path, final_image)
            print(area_dict)
            save_path = os.path.join(save_result_vis_path,file.replace('.jpg','.json'))
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(area_dict, file, ensure_ascii=False, indent=4)
            end_time = time.time() 
            elapsed_time = end_time - start_time
            print(f"The file {file} took {elapsed_time} seconds to complete.")
