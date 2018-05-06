import os
from PIL import Image
import tensorflow as tf
import numpy as np
import process_img
# WIDTH=256
# HEIGHT=256

def _read_images(img_list):
    """读取txt文件的images信息"""
    print(1)
    with open(img_list) as f:
        images=f.readlines()
        f.close()
    return images
def _parse_image(image):
    """解析一行images信息"""
    image = image.split(' ')
    imgName = image[0][:-1]#图像名
    label = image[1:-1]
    label = np.array(list(map(float, label)))
    return imgName,label

def img2TfRecord(img,label):
    """
    将image和label转换成tfrecord的example
    :param img:
    :param label: float list
    :return:
    """
    # 转换成二进制
    f_label = (label.reshape(1, -1)).tolist()[0]
    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=f_label)),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
    s_example=example.SerializeToString()
    return s_example

def createTfRecords(img_dir,filename,img_list):
    """
        根据保存图像名和label的txt文件生成tfrecords文件

        :param img_dir: 存储图像路径
        :param filename: 要生成tfrecords文件名
        :param img_list: 保存图像的list
    """
    writer=tf.python_io.TFRecordWriter(filename)
    images=_read_images(img_list)
    #imgNum=len(images)
    i=1
    for image in images:
        print(i)
        #解析image获得图像名和label
        imgName,label=_parse_image(image)
        or_image=Image.open(os.path.join(img_dir,imgName))
        #从图像中裁剪出人
        img,label=process_img.crop_data(or_image,label)
        #print(real_image)
        #将图像转换成256*256
        re_image,re_label=process_img.resize_image(img,label)
        #print(re_image.size)
        #real_image=re_image.resize((WIDTH,HEIGHT),Image.ANTIALIAS)

        #将image和label转换成example
        example=img2TfRecord(re_image,re_label)
        writer.write(example)
        i+=1

    writer.close()


createTfRecords('E:\数据集\MPII\mpii_human_pose_v1\images','test.tfrecords','../train1.txt')




