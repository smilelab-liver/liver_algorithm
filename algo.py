# OPENSLIDE_PATH = r"C:\Users\Jimmy\anaconda3\Library\openslide-win64-20231011\bin"
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
# os.add_dll_directory(OPENSLIDE_PATH)
import sys
import openslide
import cv2
# import matplotlib
# matplotlib.use('TKAgg')
# from matplotlib import pyplot as plt
import numpy as np
import xml.etree.cElementTree as ET
import openslide
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import json

from utility.utility import *
from utility.post_process import *

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
                color = color_map.get(class_idx+3, [255, 255, 255]) 
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

    # color_map = {
    #     0: [255, 255, 255],
    #     1: [0, 0, 255],
    #     2: [0, 255, 0],
    # }
    color_map = {
        0: [255, 255, 255],  # 白色
        1: [0, 0, 255],      # 红色
        2: [0, 255, 0],      # 绿色
        3: [255, 0, 0],      # 蓝色
        4: [0, 255, 255],    # 黄色
        5: [255, 0, 255],    # 品红
        6: [255, 255, 0],    # 青色
    }
    assign_bridging(lower_or_right_mask, real_mask, bboxes, wsi_img, area_dict, color_map)
    assign_bridging(upper_or_left_mask, real_mask, bboxes, wsi_img, area_dict, color_map)

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


def check_bridge_when_no_vein(fibrosis_mask):
    """
    判定是不是 Bridging
    Rule 1 : Bridging 一定會過組織中線
    Rule 2 : Bridging 會從組織邊緣生長出來 (但有可能不是有兩邊)
    Rule 2 : Bridging 一定會分布在組織中線的兩側
    """
    global tissue_contour_mask
    global wsi_data
    global wsi_level
    global tissue_skeleton


    mid_intersection = cv2.bitwise_and(tissue_skeleton, fibrosis_mask)
    crosses_middle = np.any(mid_intersection > 0)

    # 未穿過組織中線
    if not crosses_middle:
        return False
    
    tissue_contours_intersection = cv2.bitwise_and(tissue_contour_mask, fibrosis_mask)
    num_labels, labels = cv2.connectedComponents(tissue_contours_intersection)
    # 未與兩側有交集
    if num_labels - 1 < 2:
        # print("No Bridging - Insufficient regions")
        return False
    
    return True

    # # 1. 若直接與中線交集則直接判定 bridging
    # if cv2.countNonZero(result) > 0:
    #     print("Bridging detected - Tissue regions on tissue mid")
    #     return True
    
    # # 在 Caseviewer 上計算穿刺寬度大概落在 5XX~7XX 微米
    # bridge_distance_threshold_level0 = 500

    # # 0. 計算當前 level 的距離閥值
    # mpp_x = float(wsi_data.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
    # bridge_distance_threshold = int(bridge_distance_threshold_level0 / (mpp_x * wsi_data.level_downsamples[wsi_level]))

    # # 1. 計算纖維與組織邊界的交集
    # result = cv2.bitwise_and(tissue_contour_mask, fibrosis_mask)
    # num_labels, labels = cv2.connectedComponents(result)

    # if num_labels - 1 < 2:
    #     print("No Bridging - Insufficient regions")
    #     return False

    # # 2. 提取交集區域的中心點
    # centers = []
    # for label in range(1, num_labels):
    #     component_points = np.column_stack(np.where(labels == label))
    #     center = np.mean(component_points, axis=0)
    #     centers.append(center)

    # if len(centers) < 2:
    #     print("No Bridging - Insufficient centers")
    #     return False

    # # 3. 判斷是否存在兩個 Component 區域明顯分離的情況
    # centers = np.array(centers)
    # x_coords = centers[:, 1] 
    # y_coords = centers[:, 0] 

    # x_range = np.max(x_coords) - np.min(x_coords)
    # y_range = np.max(y_coords) - np.min(y_coords)

    # if x_range > bridge_distance_threshold or y_range > bridge_distance_threshold:
    #     print("Bridging detected - Tissue regions on opposite sides")
    #     return True
    # else:
    #     print("No Bridging - Tissue regions not separated")
    #     return False

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

def process_component(component_label, labels,fibrosis_mask ,fibrosis_dilated, bboxes, wsi_img, area_dict):
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

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
    overlap_bboxex = is_component_in_bbox(component_points, bboxes, class_filter={1, 2})
    # overlap_duct_bboxes = is_component_in_bbox(component_points, bboxes, class_filter={0})

    with lock:
        # 如果 BBOX 重疊就不用特別作 bridging
        # TODO: BBOX 距離太近的情況，要合併
        overlap_bboxex = check_boxes(overlap_bboxex)
        # 計算面積 & 視覺化纖維
        if len(overlap_bboxex) == 1:
            # 一個血管的情況
            x1, y1, x2, y2, class_idx, bbox_id = overlap_bboxex[0]
            wsi_img[real_component_mask > 0] = colors[class_idx] # Portal:紅色 Central:綠色
            if area_dict.get(bbox_id) is None:
                area_dict[bbox_id] = 0
            area_dict[bbox_id] += area
        elif len(overlap_bboxex) == 2:
            # TODO: 兩個血管的情況 => Briding
            cutBridge_v2(component_mask, real_component_mask, wsi_img, overlap_bboxex, area_dict)
        elif len(overlap_bboxex) > 2:
            wsi_img[real_component_mask > 0] = [0, 255, 255]  # 黄色
        else:
            # 判定 Bridging
            if area >= 1000 and check_bridge_when_no_vein(component_mask):
                wsi_img[real_component_mask > 0] = [255, 255, 0] 
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

def main_processing(labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, num_labels, area_dict):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_component, label, labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, area_dict)
            for label in range(1, num_labels)
        ]
        for future in futures:
            future.result() 
    draw_all_boxes(wsi_img, bboxes)
    return wsi_img

if __name__ == "__main__":
    global wsi_data # 要抓 wsi 的 header 資訊
    global wsi_level
    global tissue_skeleton
    global tissue_contour_mask
    wsi_level = 4
    if len(sys.argv) > 1:
        try:
            level = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer for level.")
            sys.exit(1)
    else:
        level = 3 
    json_root_path = f'/work/u3516703/liver_score/f{level}_bbox_json'
    wsi_root_path = f'/work/u3516703/liver_score/f{level}'
    xml_root_path = f'/work/u3516703/liver_score/f{level}_liver_xml'
    tissue_root_xml_path = f'/work/u3516703/liver_score/f{level}_tissue_xml/biopsy'
    save_result_vis_path = f'/work/u3516703/liver_score/f{level}_result'

    for file in os.listdir(wsi_root_path):
        if '.mrxs' in file:
            print(f"The file {file} is being processed.")
            start_time = time.time()
            wsi_path = os.path.join(wsi_root_path,file)
            json_path = os.path.join(json_root_path,file.replace('.mrxs','.geojson'))
            xml_path = os.path.join(xml_root_path,file.replace('.mrxs','.xml'))
            tissue_xml_path = os.path.join(tissue_root_xml_path,file.replace('.mrxs','.xml'))
            wsi_data = openslide.OpenSlide(wsi_path)

            # 0. 產生後處理會用到的 Mask
            bbox_mask,bboxes = generate_bbox_mask(wsi_data, json_path,level=wsi_level)
            fibrosis_mask = generate_fibrosis_mask(wsi_data, xml_path,level=wsi_level)
            tissue_mask = generate_tissue_mask(wsi_data, tissue_xml_path,level=wsi_level)
            wsi_img = generate_wsi_img(wsi_data, level=wsi_level)

            # 過濾 bbox (只留下穿刺區域的 bbox)
            assert len(bboxes) > 0, "bbox mask is empty"
            bboxes = is_component_in_bbox(np.argwhere(tissue_mask > 0).tolist(), bboxes, class_filter=None)
            assert len(bboxes) > 0, "bbox mask is empty 2"
            bboxes = generate_bbox_uid(bboxes)
            assert len(bboxes) > 0, "bbox mask is empty 3"
            
            tissue_contours, _ = cv2.findContours(tissue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tissue_contour_mask = np.zeros_like(tissue_mask)
            min_area = 1000
            for contour in tissue_contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    cv2.drawContours(tissue_contour_mask, [contour], -1, (255), thickness=10)

            # 1. 只留穿刺的纖維，並去除極值
            fibrosis_mask = cv2.bitwise_and(fibrosis_mask,tissue_mask)
            # fibrosis_mask = removeedge(fibrosis_mask)

            # 2. 將纖維的 mask 膨脹，並以 Compenent 個別處理
            fibrosis_dilated = cv2.dilate(fibrosis_mask, np.ones((9, 9), np.uint8), iterations=1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fibrosis_dilated)

            # 3. 產生組織的中間線
            tissue_skeleton = generate_skeleton(tissue_mask)

            # 4. 開始分區
            lock = threading.Lock()
            area_dict = {}
            final_image = main_processing(labels, fibrosis_mask, fibrosis_dilated, bboxes, wsi_img, num_labels, area_dict)

            # 5. 儲存結果
            if not os.path.exists(save_result_vis_path):
                os.makedirs(save_result_vis_path)
            save_path = os.path.join(save_result_vis_path,file.replace('.mrxs','.jpg'))
            final_image[tissue_skeleton > 0] = [168, 0, 121]
            cv2.imwrite(save_path, final_image)
            print(area_dict)
            save_path = os.path.join(save_result_vis_path,file.replace('.mrxs','.json'))
            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(area_dict, json_file, ensure_ascii=False, indent=4)

            end_time = time.time() 
            elapsed_time = end_time - start_time
            print(f"The file {file} took {elapsed_time} seconds to complete.")