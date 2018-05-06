# import csv
# result_list=list()
# result_list.append(['lw',1,'b',2])
# result_list.append(['s','w'])
# print(result_list)
# train_file=open('./result/trai.csv','w',newline='')
#
# train_writer=csv.writer(train_file,dialect='excel')
#
# for raw in result_list:
#     train_writer.writerow(raw)


# import tensorflow as tf
# import load_batch_data
# import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import  Image
from pylab import *
# a=np.array([[1,2],[3,4],[5,6]])
# b=np.ones((3,2))
# print(a-b)
# a=[[2,3],[2,3],[3,3]]
# b=np.array(a)
# print(b.shape)
# path=r'/home/weic/project/linux/image/'
# names=os.listdir(path)
# print(names)
# num=len(names)
# print(num)
# images=np.ndarray(shape=[num,256,256,3],dtype=np.float32)
# print(images.shape)
# img_size=[]
# for name in names:
#     img_path=os.path.join(path,name)
#     a=Image.open(img_path)
#     img_size.append(a.size)
#     image = a.resize((256, 256), Image.ANTIALIAS)
#     image.save('./testimage/%d.jpg'%(names.index(name)))

#
#     images[names.index(name),:,:,:]=np.array(image)
#     #images=np.expand_dims(image,0)
# for i in range(num):
#     plt.imshow(images[i])
#     plt.show()
#
# names = os.listdir('./testimage')
# result=[(i,os.stat(i).st_mtime) for i in (os.path.join('./testimage', name) for name in names)]
# print(result)
# result.sort()
#
# print(len(names))
# print(result)
# for name in names:
#     testimage = Image.open(os.path.join('./testimage', name))
#
#     plt.imshow(testimage)
#     plt.show()

# path=r'/home/weic/project/linux/image/'
# names=os.listdir(path)
# num=len(names)
# #images = np.ndarray(shape=[num, 256, 256, 3])
# img_size=[]
# import tensorflow as tf
# for name in names:
#     img_path = os.path.join(path, name)
#     a = Image.open(img_path)
#     na=np.array(a)
#     print(na.shape)
#     img_size.append(a.size)
#     xna=np.expand_dims(a,axis=0)
#     xna=tf.cast(xna,tf.float32)
#     #xna=np.array(xna,dtype=np.float32)
#     #xna=np.array(a)
#     print(xna.dtype)
#
# print(img_size)
from PIL import Image
path = r'/home/weic/project/linux/image/'
names = os.listdir(path)
images=[]
results=[]
for name in names:
    img_path = os.path.join(path, name)
    a = Image.open(img_path)

    expend_a = np.expand_dims(a, axis=0)
    expend_a = np.array(expend_a, dtype=np.float32)
    images.append(expend_a)
for i in range(len(images)):
    img_path = os.path.join(path, names[i])
    a = Image.open(img_path)
    a.show()
