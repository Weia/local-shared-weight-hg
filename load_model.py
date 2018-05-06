import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import load_batch_data
import os
import resnet_model
#加载模型，进行预测

#问题：把数据准备放到session外面

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = False
allx=[]
ally=[]
with tf.Session(config=config) as sess1:

    ckpt = tf.train.get_checkpoint_state('./model/')  # 通过检查文件锁定最新模型,时间
    if ckpt and ckpt.model_checkpoint_path:#ckpt.model_checkpoint_path最新的模型
        new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')  # 载入图结构
        new_saver.restore(sess1,ckpt.model_checkpoint_path)

        # 测试一张图片，问题：基本轮廓相似，但是对于图片的处理裁剪、resize、没做，对应不上到原图中
        # path=r'/home/weic/project/linux/image/'
        # names=os.listdir(path)
        # num=len(names)
        # images = np.ndarray(shape=[num, 256, 256, 3])
        # img_size=[]
        #
        # for name in names:
        #     img_path = os.path.join(path, name)
        #     a = Image.open(img_path)
        #     img_size.append(a.size)
        #     image = a.resize((256, 256), Image.ANTIALIAS)
        #
        #     images[names.index(name), :, :, :] = np.array(image)
        #
        #

        graph=tf.get_default_graph()
        # writer = tf.summary.FileWriter('load', graph=graph)
        # writer.flush()
        batch_images, batch_labels = load_batch_data.batch_samples(1, '/home/weic/project/linux/train.tfrecords', True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess1)
        for ite in range(100):
            print(ite)
            images, labels = sess1.run([batch_images, batch_labels])
            #打印所有变量
            # for op in graph.get_operations():
            #     print(op.name,' ')
            input=graph.get_tensor_by_name('input/input_images:0')
            # loss=graph.get_tensor_by_name('loss/cross_entropy_loss:0')
            output=graph.get_tensor_by_name('inference/output:0')
            test=sess1.run(output,feed_dict={input:images})
            #img=np.sum(test[0],axis=0)
            # plt.matshow(img)
            # plt.show()
            (row,col,dep)=test[0].shape

            for j in range(1):
                #print(j)
                x = []
                y = []
                for i in range(16):
                    #print('i',i)
                    index=np.argmax(test[j][i])
                    m,n=divmod(index,col)
                    #print(m,n)
                    #print(m*256/64,n*256/64)
                    #ox,oy=img_size[j]
                    x.append(m*256/64)
                    y.append(n*256/64)
                allx.append(x)
                ally.append(y)
                img=Image.fromarray(images[j])
                img.save('./testimage/%d.jpg'%(ite))
                #plt.axis([0,ox,oy,0])
                # plt.imshow(images[j])
                # plt.plot(y,x,'r*')
                # plt.show()
        coord.request_stop()
        coord.join(threads)

    #loss=sess.run(graph.get_tensor_by_name('loss/train_loss:0'),feed_dict={'input_image':image,'labels':label})
    # print(loss)
# print(len(allx),len(ally))
# names=os.listdir('./testimage')
for name in range(100):
    testimage=Image.open(os.path.join('./testimage/%d.jpg'%name))

    plt.imshow(testimage)
    print(name)

    plt.plot(ally[name],allx[name],'r*')
    plt.show()


# # 加载模型，进行预测
# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config.gpu_options.allow_growth = False
# allx = []
# ally = []
# with tf.Session(config=config) as sess1:
#     ckpt = tf.train.get_checkpoint_state('./model/')  # 通过检查文件锁定最新模型,时间
#     if ckpt and ckpt.model_checkpoint_path:  # ckpt.model_checkpoint_path最新的模型
#         new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构
#         new_saver.restore(sess1, ckpt.model_checkpoint_path)
#
#         # 测试一张图片，问题：基本轮廓相似，但是对于图片的处理裁剪、resize、没做，对应不上到原图中
#         path=r'/home/weic/project/linux/image/'
#         names=os.listdir(path)
#         num=len(names)
#         #images = np.ndarray(shape=[num, 256, 256, 3])
#         img_size=[]
#         graph = tf.get_default_graph()
#         flag=0
#         for name in names:
#             img_path = os.path.join(path, name)
#             a = Image.open(img_path)
#             (width,height)=a.size
#             img_size.append(a.size)
#             expend_a=np.expand_dims(a,axis=0)
#             expend_a = tf.cast(expend_a, tf.float32)
#             x = []
#             y = []
#             #input = graph.get_tensor_by_name('input/input_images:0')
#             # loss=graph.get_tensor_by_name('loss/cross_entropy_loss:0')
#             #output = graph.get_tensor_by_name('inference/output:0')
#             output=resnet_model.model(expend_a)
#             test = sess1.run(output)
#             for i in range(16):
#                 # 每个特征图的最大值
#                 index = np.argmax(test[0][i])
#                 m, n = divmod(index, width)
#                 # print(m,n)
#                 # print(m*256/64,n*256/64)
#                 # ox,oy=img_size[j]
#                 x.append(m * width / 64)
#                 y.append(n * height / 64)
#             allx.append(x)
#             ally.append(y)
#             img = Image.fromarray(expend_a[0])
#             img.save('./testimage/%d.jpg' % (flag))
