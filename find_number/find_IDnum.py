'''
身份证号码识别：
第一：
    先是模板的处理：
    读取、均值、灰度、二值化、轮廓查找、最大外截矩阵、然后存储数据
第二模块：
    识别对象处理：
    读取、均值、灰度、二值化、膨胀腐蚀操作（开与闭、礼帽与黑帽）、轮廓查找、找出最大面积（就是需要识别的目标）、
        最大外截矩阵、截取、 读出来，单独处理（读取、均值、灰度、二值化、轮廓查找、最大外截矩阵、然后存储数据）
最后进行模板匹配 处理数据 然后输出结果
'''

import cv2
import numpy as np

from show_img import imshow_img
from sort_number import sort_num,find_coordinate

############## 模板照片提取信息 ##################
#照片读取
template = cv2.imread(r'./img/template.png')

# 均值方波使图像平滑
template = cv2.blur(template,(3,3))

#灰度图处理
gray_tem = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

#二值化处理
ret,thre = cv2.threshold(gray_tem,150,255,cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV)

#轮廓查找 只找最大外接轮廓
contours,h = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# index 主要记录找出的方框坐标 寄存 等下会进行排序
#result_num 用来存已经识别出来的模板数字和所对应的值
index = []
result_num = {0:''} #我的模板图 它是1234567890 所有要把0移动到前面来

for value in contours:
    x,y,w,h  = cv2.boundingRect(value) #最大外接矩阵的坐标
    index.append((x,y,w,h))

print(np.array(index).shape) #判断是否有这么多值
n = len(index)
result = sort_num(index) #调用外部文件进行排序处理

#将排好序的数组坐标进行取出 然后在原图截取 存储
for (i,value) in enumerate(result):
    (x,y,w,h) = value
    roi = gray_tem[y:y+h,x:x+w]
    roi = cv2.resize(roi,(57,88))

    if i == n-1:
        result_num[0] = roi
        continue
    result_num[i+1] = roi

print('result_num',len(result_num))


################ 识别图的处理 #################
#卷积设置
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
rect_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))

img = cv2.imread(r'./img/3.jpg')
#均值滤波 去除干扰、杂质
blur = cv2.blur(img,(5,5))
#灰度处理
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值化 cv2.THRESH_OTSU为自动匹配二值化值的大小
thre_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#黑帽操作 闭（先膨，后腐）-原始图像
morp_b = cv2.morphologyEx(thre_img,cv2.MORPH_BLACKHAT,sq_kernel)

#再加个开运算（先腐，再膨胀）
next_morp = cv2.morphologyEx(morp_b,cv2.MORPH_OPEN,(31,31))

#laplacian算子
lap = cv2.Laplacian(next_morp,cv2.CV_64F)
lap = cv2.convertScaleAbs(lap)

#闭运算
morp_close = cv2.morphologyEx(lap,cv2.MORPH_CLOSE,rect_kernel2)

#轮廓查找
cnt,h = cv2.findContours(morp_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#找出最大的面积 就是我们需要识别的目标
max_area = cv2.contourArea(cnt[0])
index_num = 0
for i,value in enumerate(cnt):
    # print(cv2.contourArea(value))
    if cv2.contourArea(value) > max_area:
        max_area = cv2.contourArea(value)
        index_num = i

print(max_area,index_num)
x,y,w,h = cv2.boundingRect(cnt[index_num]) #找出外接矩阵 然后取出进行下面操作
result_x,result_y,result_w,result_h = x,y,w,h


#图像截取
result_img = img[y-2:y+h+2,x-2:x+w+2]
imshow_img('result_img',result_img)
#灰度处理
gray_result_img = cv2.cvtColor(result_img,cv2.COLOR_BGR2GRAY)
#二值化
thre_result_img = cv2.threshold(gray_result_img,150,255,cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV)[1]
imshow_img('thre_result_img',thre_result_img)
#轮廓查找
cnt,h = cv2.findContours(thre_result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


#将找出的轮廓 找出它的坐标进行排序存储
loc = []
for i in cnt:
    x , y , w , h = cv2.boundingRect ( i )
    loc.append((x , y , w , h))

loc_r = sort_num(loc)
print(loc_r)
print(np.array(loc_r).shape)

#这是在原图显示识别的数字，确定他在原图的位置方法
primitive_coordinate = (result_x,result_y)
coordinate = find_coordinate(primitive_coordinate,loc_r) #调用外部所写的函数确定原图位置
print('coordinate',np.array(coordinate).shape)

'''
最后进行模板匹配，一个一个比对，根据自己所给的匹配模板进行选值。
我这里选择的是cv2.TM_CCOEFF 值越大越匹配 所有找出最大值 然后输出结果
'''
getOutput = ''
for i,val in enumerate(loc_r):
    roi_img = gray_result_img[val[1]:val[1] + val[3] , val[0]:val[0] + val[2]]
    roi_img = cv2.resize ( roi_img , (57 , 88) ) #大小设置成之前的一样的 比较更加准确

    compare_array = []

    #模板匹配
    for j,digitROI in result_num.items():
        res = cv2.matchTemplate(roi_img,digitROI,cv2.TM_CCOEFF)
        (_,compare,_,_) = cv2.minMaxLoc(res)
        compare_array.append(compare)

    print(compare_array)
    imshow_img ( 'roi' , roi_img )
    getOutput += str(np.argmax(compare_array)) #找出数字中最大的返回下标 转化为字符串
    print('coordinate[i]',coordinate[i])
    cv2.putText(img,str(np.argmax(compare_array)),coordinate[i],cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

print(len(getOutput))
print('身份证号码为：',getOutput)
imshow_img('result',img) #图像展示
