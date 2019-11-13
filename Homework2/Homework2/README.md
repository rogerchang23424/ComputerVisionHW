## 使用方法
**注意：請不要把該資料夾內的`*.npy`刪除，否則無法正確運行**
1. 需要python安裝以下的套件
```
opencv2
numpy
scipy
matplotlib
```
2. 執行此程式的方法
> 請使用python3.7版本執行
>`hw2_1.py` 是Part 1. Camera Calibration 的主程式
>執行方法
```bash=windows
python hw2_1.py
```
>`hw2_2.py` 是Part 2. Homography transformation 的主程式
>執行方法
```bash=windows
python hw2_2.py
```
3. 執行結果
`hw2_1.py` 會在Console/Terminal顯示以下結果
```
The result of P is: [[-1.20506196e-01  2.39572920e-02  3.39537143e-02 -2.80405044e-01]
 [ 8.01387391e-03 -1.96452207e-02  1.40315785e-01 -9.40731645e-01]
 [-1.32981990e-05  2.05840374e-04  1.62619634e-04 -3.11261994e-03]]
RMSE is 0.777631
The result of P is: [[ 1.11949748e-01  3.71574660e-02 -3.18913144e-02  5.04632271e-01]
 [-1.19868837e-02  1.01116840e-02 -1.35721325e-01  8.43647803e-01]
 [ 1.07432265e-04 -1.64389891e-04 -1.47596881e-04  2.81201919e-03]]
RMSE is 0.739592
Angle: 23.290160
```
Angle 是指兩台照相機拍攝的角度，上面是`data/chessboard_1.jpg` 的 ![](https://latex.codecogs.com/png.latex?P) 值與投影過的點和原始點RMSE值；下面是`data/chessboard_2.jpg` 的 ![](https://latex.codecogs.com/png.latex?P) 值與投影過的點和原始點RMSE值

`hw2_1.py` 輸出影像會放在 `./hw2_1_results/` 資料夾裡

- `image_1_RMSE=<rmse_value>.png` 存放`data/chessboard_1.jpg`投影過的點和原始點的結果圖
- `image_2_RMSE=<rmse_value>.png` 存放`data/chessboard_2.jpg`投影過的點和原始點的結果圖
- `camera_matrix.json` 是儲存`data/chessboard_1.jpg`和`data/chessboard_2.jpg`的 ![](https://latex.codecogs.com/png.latex?P) , ![](https://latex.codecogs.com/png.latex?K) , ![](https://latex.codecogs.com/png.latex?R) , ![](https://latex.codecogs.com/png.latex?t) 的矩陣數值

`hw2_2.py` 輸出影像會放在 `./hw2_2_results/` 資料夾裡

- (2-A)`A.png` 存放含兩個物件的邊框圖，兩個物件分別用黃色和洋紅色框起來
- (2-A)`B.png` 存放含一個物件的邊框圖，用黃色框起來
- (2-A)`C.png` 存放含一個物件的邊框圖，用黃色框起來
- (2-B)`imgA_forward_result.png` 是使用**forward warping**對兩個物件互相投影後的結果
- (2-B)`imgA_backward_result.png` 是使用**backward warping**對兩個物件互相投影後的結果
- (2-C)`imgB_forward_result.png` 是Image B使用**forward warping**對Image C物件投影後的結果
- (2-C)`imgB_backward_result.png` 是Image B使用**backward warping**對Image C物件投影後的結果
- (2-C)`imgC_forward_result.png` 是Image C使用**forward warping**對Image B物件投影後的結果
- (2-C)`imgC_backward_result.png` 是Image C使用**backward warping**對Image B物件投影後的結果