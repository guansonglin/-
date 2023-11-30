'''
1.单个图像展示
2.多个图像横着展示
3.多个图像竖着展示
'''
import cv2
import numpy as np

def imshow_img(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#多个图像横着展示
def hstack_img(name,*img):
    res = np.hstack((img))
    imshow_img(name,res)

#多个图像竖着展示
def vstack_img(name,*img):
    res = np.vstack((img))
    imshow_img(name,res)