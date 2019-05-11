import cv2
import numpy as np


def dilate(img):
    """
    膨胀
    :param img:
    :return:
    """
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 获得结构元素
    # 第一个参数：结构元素形状，这里是矩形
    # 第二个参数：结构元素大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 执行膨胀
    dst = cv2.dilate(binary, kernel)
    return dst

def erode(img):
    """
    腐蚀
    :param img:
    :return:
    """
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 获得结构元素
    # 第一个参数：结构元素形状，这里是矩形
    # 第二个参数：结构元素大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 执行腐蚀
    dst = cv2.erode(binary, kernel)
    return dst


def carve(path,root):

    """
    切割path图片，放在root文件夹下
    :param path:
    :param root:
    :return:
    """
    dsize = 64  # 归一化处理的图像大小,改成28*28
    img = cv2.imread(path, 0)
    img = dilate(img)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 反转颜色
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津算法二值化
    data = np.array(img)
    len_x = data.shape[0]
    len_y = data.shape[1]
    min_val = 10  # 设置最小的文字像素高度，防止切分噪音字符

    start_i = -1
    end_i = -1
    rowPairs = []  # 存放每行的起止坐标

    # 行分割

    for i in range(len_x):
        if (not data[i].all() and start_i < 0):
            start_i = i
        elif (not data[i].all()):
            end_i = i
        elif (data[i].all() and start_i >= 0):
            # print(end_i - start_i)
            if (end_i - start_i >= min_val):
                rowPairs.append((start_i, end_i))
            start_i, end_i = -1, -1

    # print(rowPairs)

    start_j = -1
    end_j = -1
    min_val_word = (rowPairs[0][1] - rowPairs[0][0]) // 4  # 最小文字像素长度
    number = 0  # 分割后保存编号
    for start, end in rowPairs:
        for j in range(len_y):
            if (not data[start: end, j].all() and start_j < 0):
                start_j = j
            elif (not data[start: end, j].all()):
                end_j = j
            elif (data[start: end, j].all() and start_j >= 0):
                if (end_j - start_j >= min_val_word):
                    tmp = data[start:end, start_j: end_j]
                    im2save = cv2.resize(tmp, (dsize, dsize))  # 归一化处理
                    im2save = erode(im2save)
                    ret, im2save = cv2.threshold(im2save, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 反转颜色
                    cv2.imwrite(root + '%d.png' % number, im2save)
                    number += 1
                    # print("%d  pic" % number, start_j, end_j, (end - start) * (end_j - start_j))
                start_j, end_j = -1, -1

    # 列分割


def predict(path,model):
    char_dict = ['介', '前', '口', '右', '后', '唱', '左', '我', '昌', '朗', '歌', '绍', '自', '诵', '转', '进', '退']
    img= cv2.imread(path,0)
    lena = img.reshape(1,64,64,1)
    lena = lena/255    #统一格式

    # model.summary()
    pre=model.predict(lena)
    result = np.argmax(pre,axis=1)
    return char_dict[result[0]]

def output_sort(paths,model):
    string = ""
    for path in paths:
        string += predict(path,model)
    if '昌' in string:
        idx = string.index('昌')
        str1 = string[0:idx-1]
        str2 = string[idx+1:]
        return str1+'唱'+str2
    else:
        return string

def camera_shot(path):
    cap = cv2.VideoCapture(0)
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(path, frame)
            break
    cap.release()
    cv2.destroyAllWindows()