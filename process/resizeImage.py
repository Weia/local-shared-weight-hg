import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import tensorflow as tf

writer=tf.python_io.TFRecordWriter('train256v0.tfrecords')
f=open('train.txt','r')
images=f.readlines()
f.close()
print(len(images))
data_dir='E:\数据集\MPII\mpii_human_pose_v1\images'
j=0
for image in images:
    print(j)
    re_label=[]#保存变换后的label

    image=image.split(' ')
    name=image[0][:-1]
    label=list(map(float,image[1:-1]))
    o_img=Image.open(os.path.join(data_dir,name))
    o_x,o_y=o_img.size#图像原始大小
    img=o_img.resize((256,256))
    re_x,re_y=img.size#图像resize后的大小
    # 记录变换后的标记下x,y
    #x=[]
    #y=[]

    #记录变换前的标记下x,y
    #ox=[]
    #oy=[]
    for i in range(len(label)):
        if i%2==0:
            l=label[i]*re_x/o_x
            #x.append(l)#后
            #ox.append(label[i])
        else:
            l=label[i]*re_y/o_y
            #y.append(l)#后
            #oy.append(label[i])
        re_label.append(l)
    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=re_label)),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
    writer.write(example.SerializeToString())
    j+=1
    #print(re_label)
    #显示变换后的标记
    #plt.plot(x,y,'r*')

    #显示变换前的标记
    #plt.plot(ox,oy,'r*')
    #plt.imshow(img)
    #plt.show()
#写入文件



