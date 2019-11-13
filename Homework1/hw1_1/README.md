# Homework 1-1

### 使用方法

1. 需要python安裝以下的套件
```
opencv2
numpy
scipy
matplotlib
```
2. 執行此程式的方法
> 請使用python3.7版本執行
```bash=windows
python [options] <path_file>
```
`path_file` 是指想要進行哈里斯邊角偵測的圖片路徑

`options` 有以下參數：
- `dot_color`選擇輸出邊角圓點的顏色，有`red`、`green`和`blue`，預設為`red`
- `dot_radius`選擇輸出邊角圓點的半徑，預設為`2`
- `sobel_edge_threshold`選擇Sobel Edge大小的threshold，預設為`20`
- `sobel_edge_colormap`選擇輸出Sobel Edge方向圖的colormap，支持`viridis`、`plasma`、`inferno`，預設為`viridis`
- `R_k`設定Response公式中的 ![](https://latex.codecogs.com/png.latex?R%3Ddet%28A%29-k%28trace%28A%29%29%5E2) 的 ![](https://latex.codecogs.com/png.latex?k) 值，預設為`0.04`
- `nms_threshold` 設定non-maximum suppression的threshold
- `nms_threshold_rate`設定non-maximum suppression與response最大值的比值作為threshold，預設為`0.01`
如果要添加選項，請用下列的方式添加
例如：
如果要將輸出的點設為綠色，那麼執行的程式為
```
python --dot_color=green <path_file>
```
3. 執行結果
程式執行結果皆會放在`./results/`的資料夾裡
- `<filename>_gaussian_k=5.png` 存放使用 ![](https://latex.codecogs.com/png.latex?%5Csigma%3D5) 及 kernel_size=5進行高斯處理後的結果圖
- `<filename>_gaussian_k=10.png` 存放使用 ![](https://latex.codecogs.com/png.latex?%5Csigma%3D5) 及 kernel_size=10進行高斯處理後的結果圖
- `<filename>_magnitude_k=5` 存放使用Gaaussion filter size為5處理後的照片，進行Sobel Edge Detection後的大小亮度圖
- `<filename>_direction_k=5` 存放使用Gaaussion filter size為5處理後的照片，進行Sobel Edge Detection後的方向彩度圖
- `<filename>_magnitude_k=10` 存放使用Gaaussion filter size為10處理後的照片，進行Sobel Edge Detection後的大小亮度圖
- `<filename>_direction_k=10` 存放使用Gaaussion filter size為10處理後的照片，進行Sobel Edge Detection後的方向彩度圖
- `<filename>_corners_w=3.png` 使用Structure Tensor的window size為3的邊角點圖
- `<filename>_corners_w=30.png` 使用Structure Tensor的window size為30的邊角圖
- `<filename>_rotate30_corners` 為圖片旋轉 ![](https://latex.codecogs.com/png.latex?30%5E%7B%5Ccirc%7D) 後的邊角圖
- `<filename>_scale0.5_corners.png` 為圖片放大 0.5 倍的邊角圖