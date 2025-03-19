# 參數纖維演算法

## 目標
我們有 segmentation model 輸出的肝臟纖維 mask，用 YOLO 偵測出的血管位置，使用血管的位置對纖維做分區計算，並偵測以及切分 bridging 纖維。

## 使用 
使用前須準備好
1. tissue segmentation 的 json 檔。
2. fibrosis segmentation 的 xml 檔。
3. vein detection 的 json 檔。
4. WSI image。

## 程式流程
1. 預處理 fibrosis 和 vein 的 mask，確保只對穿刺部份做演算法。
2. 將纖維 dialate ，並轉成 component 做後續計算。
3. 用 `generate_skeleton` 產生組織的中線，供我們後續演算法使用。
4. 用 `main_processing`，開始演算法。
5. 儲存結果，會儲存可視化圖片，以及分區參數的 json 檔。

## 演算法設計
分成三個 case 處理
Case 1 : Only 1 vein overlap
Case 2 : At least 2 veins overlap => Bridging 分割的演算法
Case 3 : No Vein overlap => Bridging 判定的演算法
