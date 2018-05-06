import tensorflow as tf
import gene_hm
import numpy as np
"""加载一个batchsize的image"""
WIDTH=256
HEIGHT=256
HM_HEIGHT=64
HM_WIDTH=64
def _read_single_sample(samples_dir):

    filename_quene=tf.train.string_input_producer([samples_dir])
    reader=tf.TFRecordReader()
    _,serialize_example=reader.read(filename_quene)
    features=tf.parse_single_example(
        serialize_example,
        features={
                    'label':tf.FixedLenFeature([32],tf.float32),
                    'image':tf.FixedLenFeature([],tf.string)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [HEIGHT,WIDTH, 3])#！reshape 先列后行
    label = tf.cast(features['label'], tf.float32)
    return image,label
    # print(img.shape)
    # print(label)


def batch_samples(batch_size,filename,shuffle=False):
    """
    filename:tfrecord文件名
    """

    image,label=_read_single_sample(filename)
    label=tf.reshape(label,[-1,2])
    label = gene_hm.resize_label(label)#将label放缩到64*64
    #label=gene_hm.tf_generate_hm(HM_HEIGHT, HM_WIDTH ,label, 64)
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, min_after_dequeue=1000,num_threads=2,capacity=2000)
    else:
        image_batch,label_batch=tf.train.batch([image,label],batch_size, num_threads=2)

    return image_batch,label_batch



# # # # """测试加载图像"""
# import matplotlib.pyplot as plt
# #import load_batch_data
# # from PIL import Image
# #import numpy as np
# #from pyheatmap import HeatMap
# from  pyheatmap.heatmap import HeatMap
# #import HeatMap
# # with tf.Session() as sess:
# #     init_op = tf.global_variables_initializer()
# #     sess.run(init_op)
# #     image,label=load_batch_samples(23,sess,'./process/train256v3.tfrecords')
# #     #assert np.array_equal(label,np.zeros([3, 16, 64, 64],dtype=np.float32)),'error'
# #     for i in range(23):
# #         img=Image.fromarray(image[i], 'RGB')#这里Image是之前提到的
# #
# #         l=np.sum(label[i],axis=0)
# #         #assert np.array_equal(l, np.zeros([64, 64])), 'error'
# #         #print(l[30:35,53:58])
# #
# #         #plt.matshow(l, fignum=0)
# #         plt.imshow(img,cmap='Greys_r')
# #
# #         plt.show()
# #
# with tf.Session() as sess: #开始一个会话
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     # image,label=read_single_sample('test.tfrecords')
#     image_batch,label_batch=batch_samples(15,'test.tfrecords')
#     #image_batch,label_batch=tf.train.batch([image,label], batch_size=3, capacity=200, num_threads=2)
#
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#
#
#
#     for i in range(2):
#         example, l = sess.run([image_batch, label_batch])  # 在会话中取出image和label
#         print("1",l.shape)
#
#
#         for i in range(15):
#             #img=Image.fromarray(example[i], 'RGB')#这里Image是之前提到的
#             #img.save('./testimg'+str(i)+'.jpg')#存下图片
#             #print(l[i])
#
#             #label = gene_hm.generate_hm(HM_HEIGHT, HM_WIDTH, l[i], 64)
#             #print(label.shape)
#             #label=np.sum(l[i],axis=0)
#             #print(label.shape)
#             #print(label)
#             #x=l[i][:,0]
#             #y=l[i][:,1]
#             #print(img)
#             # img = img.convert('L')
#             # img=np.array(img)/255
#
#             #print(example[i])
#             #print(img)
#             #plt.matshow(label, fignum=0)
#             plt.imshow(example[i],cmap='Greys_r')
#             #plt.plot(x,y,'r*')
#
#             #print(len(label.tolist()))
#             #hm=HeatMap(label)
#             #for i in range(16):
#
#             plt.show()
# #     #print(example, l)
#     coord.request_stop()
#     coord.join(threads)
