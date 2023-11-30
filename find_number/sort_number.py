'''
根据数组 排序大小
这里采用冒泡排序
（你们也可以采用其他排序 快速、插入、堆、猴子等排序）
'''

#排序
def sort_num(value):
    n = len(value)
    for i in range(n-1):
        min_num = value[i][0]
        for j in range(i+1,n):
            if min_num > value[j][0]:
                min_num = value[j][0]
                value[i],value[j] = value[j],value[i]

    return value

#原始图像的坐标处理
def find_coordinate(primitive,value):
    result = [primitive]
    n = len(value)
    for i in range(1,n):
        result.append((result[i-1][0]+(value[i][0]-value[i-1][0]),primitive[1]))

    return result
