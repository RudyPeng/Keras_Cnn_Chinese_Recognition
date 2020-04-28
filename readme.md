# 从零开始 Kears CNN汉字识别(17个汉字)
## 拒绝烂尾，从我做起--2020.4.28上传模型（很早之前写的代码，不是很漂亮，没什么时间去整理）

### 数据准备
__data__ 文件夹下共有2个子文件夹： __test__ 和 __train__ 存放训练和测试使用的照片，共训练17个汉字，总计2000余张手写汉字照片。

[点此下载模型，提取码：1irh](链接：https://pan.baidu.com/s/1xI_QCvK_a4_CoqlNG3IWoQ)

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
 
 ### 注意事项
 如果要进行测试的话，主目录下已经保存了一张 __shot.png__ 用来测试，可以注释掉主函数中的
` f.camera_shot(path)`  运行测试查看结果  

要预测别的文字的话可以将图片放在 __train__ 的子目录下，并且修改`Training.py`中的相关参数以及
`predict`函数中的char_dict即可，具体可以参见各文件中的注释

 __output_sort函数主要是为了能将可能被切割开的‘唱’字进行合并而已，暂时没有做别的解决方法__


[Kears代码来源][1]
[手写汉字图片源自模式识别国家重点实验室][2]

[1]:https://blog.csdn.net/codebay118/article/details/72630091

[2]:http://www.nlpr.ia.ac.cn/cn/index.html
