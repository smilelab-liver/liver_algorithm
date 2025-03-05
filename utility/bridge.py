import cv2
import numpy as np
from plantcv import plantcv as pcv
import math
from utility.post_process import is_component_in_bbox , generate_distance_transform

def fillhole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:  # 只填充内部的洞（有父轮廓）
            cv2.drawContours(mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
    return mask

def fillhole2(mask):
    h, w = mask.shape
    filled_mask = mask.copy()
    flood_fill_mask = np.zeros((h+2, w+2), np.uint8)

    # 使用 floodFill 填充外部背景
    cv2.floodFill(filled_mask, flood_fill_mask, (0,0), 255)

    # 取反，得到内部的孔洞
    holes = cv2.bitwise_not(filled_mask)
    result = cv2.bitwise_or(mask, holes)
    return result



def generate_Bboxmask(mask, bboxes):
    # List of bounding boxes [(x1, y1, x2, y2, class_idx, bbox_id), ...] (至少 2 bboxes)
    bbox_mask = np.zeros_like(mask)
    for (x1, y1, x2, y2, class_idx, bbox_id) in bboxes:
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
    return bbox_mask

def find_circles_intersection(mask, bboxes):
    center = []
    # Get Center
    for (x1, y1, x2, y2, class_idx, bbox_id) in bboxes:
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
        center.append((x, y))
    radius = []
    # Get Distance
    for i, box1 in enumerate(center):
        x1, y1 = box1
        for j, box2 in enumerate(center[i+1:]):
            x2, y2 = box2
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            radius.append((i, i+j+1, distance))
    # Sort by distance
    radius.sort(key=lambda x: x[2])
    radius = radius[:len(bboxes)-1]

    # Circle mask
    intersection_mask = np.zeros_like(mask)
    for (box_idx1, box_idx2, distance) in radius:
        temp1 = np.zeros_like(mask)
        cv2.circle(temp1, center[box_idx1], int(distance), 255, -1)
        temp2 = np.zeros_like(mask)
        cv2.circle(temp2, center[box_idx2], int(distance), 255, -1)
        temp1 = cv2.bitwise_and(temp1, temp2)
        intersection_mask = cv2.bitwise_or(intersection_mask, temp1)
    return intersection_mask

def find_smallest_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (cx, cy), (w, h), angle = rect
        if w < h:
            w, h = h, w
            angle += 90
            box = np.roll(box, shift=-1, axis=0)

        mid_point1 = ((box[1] + box[2]) // 2).tolist()
        mid_point2 = ((box[0] + box[3]) // 2).tolist()
        res.append((mid_point1, mid_point2))
    return res

def split_by_line(mask, mid_point1, mid_point2):
    x1, y1 = mid_point1
    x2, y2 = mid_point2

    # 計算直線方程式 y = mx + b
    if x2 - x1 == 0:
        m = None  # 垂直線
        b = x1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

    # 獲取目前 mask 內的分類數量
    unique_classes = np.unique(mask)
    cut_classes = len(unique_classes) - 1

    # 建立一個新的 mask 來標記線條
    line_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.line(line_mask, mid_point1, mid_point2, 255, 2)

    # 找到與線條有交集的 class
    target_class = None
    for class_idx in unique_classes:
        if class_idx == 0:
            continue
        class_mask = (mask == class_idx).astype(np.uint8)
        intersection = cv2.bitwise_and(class_mask, line_mask)
        if np.count_nonzero(intersection) != 0:
            target_class = class_idx

    # 如果沒有交集，則返回空
    if target_class is None:
        return np.zeros_like(mask), np.zeros_like(mask), None

    # 取得要切割的區域
    mask = (mask == target_class).astype(np.uint8)

    # 產生網格座標 (vectorized)
    h, w = mask.shape
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # 初始化左右 mask
    left_mask = np.zeros_like(mask, dtype=np.uint8)
    right_mask = np.zeros_like(mask, dtype=np.uint8)

    # 使用 NumPy 向量化計算區域
    if m is None:
        # 垂直線 (x < b 在左邊, x > b 在右邊)
        left_mask[(mask > 0) & (x_indices < b)] = 255
        right_mask[(mask > 0) & (x_indices >= b)] = 255
    else:
        # 一般情況: y < mx + b 在左側, 其餘在右側
        left_mask[(mask > 0) & (y_indices < (m * x_indices + b))] = 255
        right_mask[(mask > 0) & (y_indices >= (m * x_indices + b))] = 255

    return left_mask, right_mask, target_class

def cut_bridge(mask, bboxes):
    """
    @mask : component mask
    @bboxes: List of bounding boxes [(x1, y1, x2, y2, class_idx, bbox_id), ...] (至少 2 bboxes)
    @return : bridge mask 值為 1, 2, 3, ... 若兩個 bbox 值 mask 就為 1, 2 分別代表兩個區域
    """
    vis = mask.copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 1. fill inner hole
    mask = fillhole2(mask)
    # 2. Find the component skeleton
    df = generate_distance_transform(mask)
    skeleton = pcv.morphology.skeletonize(df)
    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton, size=200)
    vis[pruned_skeleton > 0] = [0, 255, 0]
    # 3. Find the intersection of two circle & Use intersection to filter skeleton
    intersection_mask = find_circles_intersection(mask, bboxes)
    pruned_skeleton = cv2.bitwise_and(intersection_mask, pruned_skeleton)
    # 4. Filter skeleton if skeleton points in bboxes
    bbox_mask = generate_Bboxmask(mask, bboxes)
    bbox_mask = cv2.bitwise_and(bbox_mask, pruned_skeleton)
    pruned_skeleton[bbox_mask > 0] = 0
    # 5. find the bridge skeleton (overlap with 2 bboxes)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(pruned_skeleton, connectivity=8)
    for label in range(1, num_labels):
        sk_mask = np.zeros_like(mask)
        sk_mask[labels == label] = 255
        sk_points = np.argwhere(sk_mask > 0).tolist()
        overlaps = is_component_in_bbox(sk_points, bboxes)
        if len(overlaps) != 2:
            pruned_skeleton[sk_mask > 0] = 0
    # 6. Find the smallest rectangle & find the 2 middle points
    midpoints = find_smallest_bbox(pruned_skeleton)
    # 7. use 2 middle points to create line & distinguish bridge
    mask[mask > 0] = 1
    for (mid_point1, mid_point2) in midpoints:
        left_mask, right_mask, target_class = split_by_line(mask, mid_point1, mid_point2)
        new_class = len(np.unique(mask))
        mask[left_mask > 0] = target_class
        mask[right_mask > 0] = new_class
        cv2.line(vis, mid_point1, mid_point2, [0, 255, 0], 2)
        cv2.imwrite('vis.png', vis)
    return mask