import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import load_batch_data
import os
import resnet_model

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

pl_image=tf.placeholder(tf.float32,name='pl_input')
with tf.name_scope('inference'):
    output=resnet_model.model(pl_image)

with tf.Session() as sess1:
    ckpt=tf.train.get_checkpoint_state('./model/')
    #print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver=tf.train.Saver()
        saver.restore(sess1,ckpt.model_checkpoint_path)
        #print(ckpt.model_checkpoint_path)
        for i in range(len(images)):
            print(images[i].shape)
            result=sess1.run(output,feed_dict={pl_image:images[i]})
            results.append(result)
    else:
        print('fail to load model')
print(len(results))

for j in range(len(results)):
    print(results[0].shape[2],results[0].shape[3])
    img_path = os.path.join(path, names[j])
    img = Image.open(img_path)
    x=[]
    y=[]
    (width,height)=img.size
    print((width,height))
    for i in range(16):
        # 每个特征图的最大值

        index = np.argmax(results[0][0][i])
        print(index)
        m, n = divmod(index, (results[0].shape[2]))
        print(m,n)
        # print(m*256/64,n*256/64)
        # ox,oy=img_size[j]
        x.append(m * width / (results[0].shape[2]))
        y.append(n * height / (results[0].shape[3]))
    #plt.matshow(np.sum(results[0][0],axis=0))
    plt.imshow(img)
    plt.plot(y,x,'r+')
    plt.show()

