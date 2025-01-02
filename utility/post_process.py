#                        ____________
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#  _____________________|            |_____________________
# |                                                        |
# |                                                        |
# |                                                        |
# |_____________________              _____________________|
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |            |
#                       |____________|
#  .............................................  
#                  此檔案為 Specific Utility

import cv2
import numpy as np
from plantcv import plantcv as pcv

###################################################################
# Generate Skeleton
###################################################################
def generate_distance_transform(mask):
    """
    用距離轉換找出組織的中線，
    類似於細化組織
    """
    mask = mask.copy()
    mask = (mask > 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, tissue_dt = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    tissue_dt = np.uint8(tissue_dt)
    return tissue_dt

def generate_skeleton(mask):
    """
    透過距離轉化計算組織的中線 (降低運算)
    並找出組織的骨架，移除分支
    """
    mask = generate_distance_transform(mask)
    skeleton = pcv.morphology.skeletonize(mask=mask)
    pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=200)
    return pruned_skeleton

###################################################################
# BBOX 處理
###################################################################
class UniqueIDGenerator:
    def __init__(self):
        self.portal_count = 0
        self.central_count = 0
        self.duct_count = 0

    def get_unique_id(self, key_type):
        """
        生成 unique ID
        """
        if key_type == 'portal':
            unique_id = f"portal#{self.portal_count}"
            self.portal_count += 1
        elif key_type == 'central':
            unique_id = f"central#{self.central_count}"
            self.central_count += 1
        elif key_type == 'duct':
            unique_id = f"duct#{self.duct_count}"
            self.duct_count += 1
        else:
            raise ValueError("Invalid key type. Must be 'portal' or 'central'.")
        return unique_id
    
def draw_all_boxes(wsi_img, bboxes):
    """
    將所有偵測出的 bbox 畫在圖上
    @param bboxes: [(x1, y1, x2, y2, class_idx, bbox_id), ...]
    """
    bbox_colors = [(255, 0, 0), (255, 255, 0), (127, 0, 255)]
    # 畫 BBox
    for bbox in bboxes:
        x1, y1, x2, y2, class_idx, bbox_id = bbox
        if class_idx != 0:
            cv2.rectangle(wsi_img, (x1, y1), (x2, y2), bbox_colors[class_idx], 2)
            (text_w, text_h), baseline = cv2.getTextSize(bbox_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(wsi_img, (int(x1), int(y1) - text_h - baseline), (int(x1) + text_w, int(y1)), bbox_colors[class_idx], -1)
            cv2.putText(wsi_img, bbox_id, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(wsi_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

def generate_bbox_uid(bboxes):
    """
    產生 bbox 的 unique ID
    @param bboxes: List of bounding boxes [(x1, y1, x2, y2, class_idx, bbox_id), ...]
    @return bboxes: List of bounding boxes [(x1, y1, x2, y2, class_idx, bbox_id), ...]
    """
    generator = UniqueIDGenerator()
    class_name = ['duct', 'portal', 'central']
    res = []
    for idx, (x1, y1, x2, y2, class_idx, _) in enumerate(bboxes):
        bbox_id = generator.get_unique_id(class_name[class_idx])
        res.append((x1, y1, x2, y2, class_idx, bbox_id))
    return res

def is_component_in_bbox(component_points, bboxes, class_filter=None):
    """
    檢查 component 的所有點是否在任意 bbox 範圍內
    @param component_points: list of [x, y]
    @param bboxes: [(x1, y1, x2, y2, class_idx, bbox_id), ...]
    @param class_filter: 篩選 bbox 條件 (list 或 set)。
      - 如果為 None => 則不篩選任何 bbox。
      - 如果為 list 或 set => 僅檢查屬於該 class 的 bbox。
    @return overlap_bbox: 與 component 有重疊的 bbox 。
    """
    overlap_bbox = []
    component_points = {tuple(point) for point in component_points}
    for idx, (x1, y1, x2, y2, class_idx, bbox_id) in enumerate(bboxes):
        if class_filter is not None and class_idx not in class_filter:
            continue
        bbox_points_set = {(y, x) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)}
        if any(point in bbox_points_set for point in component_points):
            overlap_bbox.append(bboxes[idx])
    return overlap_bbox

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