from keras.models import load_model
import cv2
import numpy as np


char_dict=['介','前','口','右','后','唱','左','我','昌','朗','歌','绍','自','诵','转','进','退']

path = "./fg/fg0.png"
img= cv2.imread(path,0)
lena = img.reshape(1,64,64,1)
lena = lena/255    #统一格式
model = load_model("model2.h5")   #加载模型
# model.summary()
pre=model.predict(lena)

result = np.argmax(pre,axis=1)
print(char_dict[result[0]],end='')