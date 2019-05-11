## Kears CNN 汉字识别

### 数据准备
在data文件夹下有2个子文件夹 __test__ 和 __train__ (手写汉字来自模式识别国家重点实验室)
共有2000多张手写汉字图片,只训练了17个汉字

### 用到的库  

- kears
- cv2
- os
- shutil
- numpy
- glob

### 具体实现
`拍照识别.py` 主函数  
`Training.py` 进行汉字图片的读取训练，并保存为model2.h5  
`单个预测.py` 测试用的预测单个图片汉字  
`img_function.py` 主函数调用的函数都在这，注意函数`output_sort`嵌套了上面的预测函数
 
 ### 注意
 如果想测试的话，主目录下已经保存了一张 __shot.png__ 用来测试，可以注释掉主函数中的
` f.camera_shot(path)`  运行测试查看结果  
要预测别的文字的话可以将图片放在 __train__ 目录下，并且修改`Training.py`中的相关参数以及
`predict`函数中的char_dict即可

#### 其实output_sort函数主要是为了能将无意中切割开的‘唱’字合并而已。。懒得弄别的解决方法


[Kears代码来源][1]

[1]:https://blog.csdn.net/codebay118/article/details/72630091