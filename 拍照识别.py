import glob
import os
import shutil
import img_function as f
from keras.models import load_model

model = load_model("model2.h5")   #加载模型

path = 'shot.png'  # 拍照保存图片
root = './fg/' # 放图片的文件夹

if __name__ == '__main__':

    #摄像头拍照，按Q键拍照并退出
    f.camera_shot(path)

    # 切割
    f.carve(path,root)

    # 文件夹下的图片进行排序
    paths = glob.glob(os.path.join(root, '*.png'))
    paths.sort()

    # 整理后输出,本函数包括预测
    predict = f.output_sort(paths,model)
    print(predict)
    # 删除切割的图片
    shutil.rmtree('fg')
    os.mkdir('fg')