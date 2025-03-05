import cv2
OPENSLIDE_PATH = r"C:\Users\Jimmy\anaconda3\Library\openslide-win64-20231011\bin"
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
os.add_dll_directory(OPENSLIDE_PATH)
import sys
import openslide
import json

"""
轉換成 qupath format (如果開 mrxs)
轉過去用 -
轉回來用 +
"""

def read_bbox_from_json(file_path, bounds_x, bounds_y):
    """
    讀取 Bounding Box 的 geojson，並對座標進行調整
    @param file_path: geojson 檔案路徑
    @param bounds_x: X 座標偏移量
    @param bounds_y: Y 座標偏移量
    @return: 調整後的 JSON
    """
    adjusted_bboxes = []

    with open(file_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

        for data in datas:
            # 確保座標為數字，並進行座標調整
            adjusted_coords = [[[int(x) + int(bounds_x), int(y) + int(bounds_y)] for x, y in coords] for coords in data["geometry"]["coordinates"]]
            
            # 只取前 4 個點（去掉重複點）
            data["geometry"]["coordinates"] = adjusted_coords

            adjusted_bboxes.append(data)

    return adjusted_bboxes



wsi_path = r'f3\17-00254-Masson.mrxs'
wsi_data = openslide.OpenSlide(wsi_path)
bounds_x = wsi_data.properties["openslide.bounds-x"]
bounds_y = wsi_data.properties["openslide.bounds-y"]
# print(bounds_x, bounds_y)
json_path = r'f3_bbox_json\17-00254-Masson.geojson'
save_path = r'f3_bbox_json\17-00254-Masson.geojson'
json_data = read_bbox_from_json(json_path, bounds_x, bounds_y)
with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)