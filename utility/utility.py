#  .............................................  
#                     _ooOoo_  
#                    o8888888o  
#                    88" . "88  
#                    (| -_- |)  
#                     O\ = /O  
#                 ____/`---'\____  
#               .   ' \\| | `.  
#                / \\||| : ||| \  
#              / _||||| -:- |||||- \  
#                | | \\\ - / | |  
#              | \_| ''\---/'' | |  
#               \ .-\__ `-` ___/-. /  
#            ___`. .' /--.--\ `. . __  
#         ."" '< `.___\_<|>_/___.' >'"".  
#        | | : `- \`.;`\ _ /`;.`/ - ` : | |  
#          \ \ `-. \_ __\ /__ _/ .-` / /  
#  ======`-.____`-.___\_____/___.-`____.-'======  
#                     `=---='  
#  .............................................  
#           佛祖保佑             永无BUG 
#           此檔案為 General Utility

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
# import matplotlib
# matplotlib.use('TKAgg')
# from matplotlib import pyplot as plt
import numpy as np
import json

###################################################################
# 讀取標註檔案
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
# 產生標註檔案專區
###################################################################
def generate_fibrosis_mask(wsi_data, xml_path,level=3):
    """
    讀取 xml 檔案，回傳 mask (GrayScale)
    """
    scale_rate = 2 ** level
    classes = {'duct':0 ,'portal':1, 'central':2}
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    
    # 1. Get Image Shape & Generate Mask
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
def generate_tissue_mask(wsi_data, xml_path, level=3, min_area_threshold=10000):
    scale_rate = 2 ** level
    classes = {'duct': 0, 'portal': 1, 'central': 2}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # 1. Get Image Shape & Generate Mask
    width, height = wsi_data.level_dimensions[level]
    mask = np.zeros((height, width), np.uint8)

    # 2. Read BBox Json
    json_datas = read_fibrosis_xml(xml_path)
    fibrosis_contours,lumen_contours = parse_json(json_datas)


    for contour in fibrosis_contours:
        coordinates = contour['coordinates']
        coordinates = [[int(value / scale_rate) for value in coordinate] for coordinate in coordinates]
        coordinates = np.array([coordinates], dtype=np.int32)
        area = cv2.contourArea(coordinates)
        if area >= min_area_threshold:
            mask = cv2.fillPoly(mask, coordinates, 255)
    return mask
def generate_bbox_mask(wsi_data, json_path,level=3):
    """
    讀取 bbox geojson，回傳 bbox mask (BGR)
    """
    scale_rate = 2 ** level
    classes = {'duct':0 ,'portal':1, 'central':2}
    colors = [(255,0,0),(0,255,0),(0,0,255)]

    # 1. Get Image Shape & Generate Mask
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
        bboxes.append((x1, y1, x2, y2, class_idx, 'test'))
    return mask,bboxes
def generate_wsi_img(wsi_data, level=3):
    width, height = wsi_data.level_dimensions[level]
    wsi_img = wsi_data.read_region((0, 0), level, (width, height))
    wsi_img = cv2.cvtColor(np.array(wsi_img), cv2.COLOR_RGB2BGR)
    return wsi_img
