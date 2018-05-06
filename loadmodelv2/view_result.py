# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:49
# @Author  : weic
# @FileName: view_result.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
with open('./result.txt') as f:
    results=f.readlines()

    for result in results:
        #print(type(result))
        #print(result)
        list_result=result.split('*')
        #print(list_result)
        imgPath=list_result[0]
        #print(list_result[1].split(' '))
        label=list(map(float,list_result[1].split(' ')[:-1]))
        print(label)
        label=np.asarray(list((map(lambda x:x*256/64,label)))).reshape(-1,2)
        x=label[:,0]
        y=label[:,1]

        print(label)
        print(x)
        print(y)
        img=Image.open(imgPath).resize((256, 256),Image.ANTIALIAS)
        plt.imshow(img)
        plt.plot(x,y,'r*')
        plt.show()

