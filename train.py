import tensorflow as tf
import os
import resnet_model
import load_batch_data
import matplotlib.pyplot as plt
# import Image
import numpy as np
import processing
import gene_hm
import time
import csv
import sys

Num_Epoch=1000
batch_size=3
filename='/media/weic/新加卷/my_dataset/final_train.tfrecords'
GPU='/gpu:0'
CPU='/cpu:0'
device=GPU
learning_rate=2.5e-4
decay_steps=2000
decay_rate=0.95
staircase=True
step_to_save=200
epochSize=4200
step_to_val=1000
valIters=10
def cal_acc(output,gtMaps,batchSize):
    #计算准确率
    def _argmax(tensor):
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err( u, v):
        u_x, u_y = _argmax(u)
        v_x, v_y =_argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                         tf.to_float(91))
    def _accur(pred, gtMap, num_image):
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err,_compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)

    joint_accur = []
    for i in range(gene_hm.nPoints):
        joint_accur.append(
            _accur(output[:, i,:, :], gtMaps[:, i,:, :],batchSize))
    return joint_accur
def train(lr):
    print(learning_rate)
    #global_step=tf.train.get_or_create_global_step()
    global_step=tf.Variable(0,trainable=False)
    print('==create model==')
    #with tf.device(device):
    with tf.name_scope('input'):
        input_image = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_images')
        labels = tf.placeholder(tf.float32, [None, 16, 64, 64], name='labels')
    print('--input done--')
    with tf.name_scope('inference'):
        logits=resnet_model.model(input_image)
    print('--inference done--')
    with tf.name_scope('loss'):
        diff=tf.subtract(logits,labels)
        train_loss=tf.reduce_mean(tf.nn.l2_loss(diff,name='l2loss'))
        #train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels= labels),name='cross_entropy_loss')
    print('--loss done--')

    #with tf.device(CPU):
    with tf.name_scope('accuracy'):
        joint_accur=cal_acc(logits,labels,batch_size)
    # with tf.name_scope('lr'):
    #     decay_lr=tf.train.exponential_decay(lr,global_step,decay_steps,decay_rate,staircase,name='learning_rate')#指数式衰减
    with tf.name_scope('saver'):
        saver=tf.train.Saver()
    #with tf.device(device):
    with tf.name_scope('train'):
        with tf.name_scope('optimizer'):
            opti=tf.train.RMSPropOptimizer(learning_rate)
        train_op=opti.minimize(train_loss,global_step=global_step)
    print('--minimize done--')
    init=tf.global_variables_initializer()
    print('--init done--')
    #with tf.device(CPU):
    tf.summary.scalar('loss',train_loss,collections=['train'])
    #tf.summary.scalar('learning_rate',decay_lr,collections=['train'])
    for i in range(gene_hm.nPoints):
        tf.summary.scalar(str(i),joint_accur[i],collections=['train','test'])

    merged_summary_train=tf.summary.merge_all('train')
    merged_summary_test=tf.summary.merge_all('test')

    train_list=list()
    val_list=list()
    train_list.append(['learning_rate',learning_rate,'training_epoch',Num_Epoch,
                        'batch_size',batch_size,
                        ])
    val_list.append(['val_step','val_loss'])
    train_list.append(['train_step','train_loss'])

    exma_list=[]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    config.gpu_options.allow_growth = True


    batch_images, batch_labels = load_batch_data.batch_samples(batch_size, filename,True)
    val_images, val_labels = load_batch_data.batch_samples(batch_size, filename, True)

    with tf.name_scope('Session'):
        #with tf.device(device):


        sess=tf.Session(config=config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        ckpt=tf.train.get_checkpoint_state('./model/')
        if ckpt and ckpt.model_checkpoint_path:
            print('load model')
            saver.restore(sess,ckpt.model_checkpoint_path )
           
        train_writer = tf.summary.FileWriter('log/train', graph=tf.get_default_graph())
        val_writer=tf.summary.FileWriter('log/val')
        print('Start train')
        with tf.name_scope('training'):
            for epoch in range(Num_Epoch):
                cost=0.0
                val_cost=0.0
                print('* * * *第%d个Epoch* * * *'%(epoch+1))
                beginTime=time.time()
                for i in range(epochSize):
                    example, l = sess.run([batch_images, batch_labels])
                    if np.any(np.isnan(example)):
                        print('no images')
                        continue
                    if np.any(np.isnan(l)):
                        print('no label')
                        continue
                    exma_list.append(example.shape)
                    nor_images = processing.image_normalization(example)  # 归一化图像
                    label = gene_hm.batch_genehm(batch_size, l)  # heatmap label


                    train_step = sess.run(global_step)
                    if (i+1)%step_to_save==0:

                        _, loss, summary = sess.run([train_op, train_loss, merged_summary_train],
                                                    feed_dict={input_image: nor_images, labels: label})

                        train_writer.add_summary(summary,train_step)

                        saver.save(sess,os.path.join(os.getcwd(),'model/model%d.ckpt'%(train_step)))
                        #print(save_path)
                    else:
                        _, loss,output = sess.run([train_op, train_loss,logits],
                                                    feed_dict={input_image: nor_images, labels: label})
                        for i in range(batch_size):
                            plt.matshow(np.sum(output[0], axis=0))
                            plt.show()
                    #csv
                    train_list.append([train_step,loss])

                    cost+=loss

                    print('第%d个batch的loss%f'%(i+1,loss))
                epoch_cost=cost/(epochSize)
                print('* *第%d个epoch的loss%f* *' % (epoch + 1, epoch_cost))
                oneEpoch = time.time()
                print('one epoch cost time:', str(oneEpoch - beginTime))
                print('* '*20)
                valBegin = time.time()
                for j in range(valIters):
                    example, l = sess.run([val_images, val_labels])
                    if np.any(np.isnan(example)):
                        print('no images')
                        continue
                    if np.any(np.isnan(l)):
                        print('no label')
                        continue
                    nor_images = processing.image_normalization(example)  # 归一化图像
                    label = gene_hm.batch_genehm(batch_size, l)  # heatmap label
                    _, v_loss = sess.run([train_op, train_loss],
                                       feed_dict={input_image: nor_images, labels: label})
                    val_summaries = sess.run(merged_summary_test, feed_dict={input_image: nor_images, labels: label})
                    val_step=epoch*10+j+1
                    val_writer.add_summary(val_summaries, val_step )

                    val_list.append([val_step,v_loss])
                    val_cost+=v_loss
                val_loss=val_cost/(valIters)


                print('val cost time:',str(time.time()-valBegin))
                print('val loss:',val_loss)
                print('* ' * 20)
                train_file = open('result/train.csv', 'w', newline='')
                val_file = open('result/val.csv', 'w', newline='')
                ex_file=open('result/ex.csv','w',newline='')
                train_csv_writer = csv.writer(train_file, dialect='excel')
                val_csv_writer = csv.writer(val_file, dialect='excel')
                ex_csv_writer=csv.writer(ex_file,dialect='excel')
                for raw in train_list:
                    train_csv_writer.writerow(raw)
                for line in val_list:
                    val_csv_writer.writerow(line)
                train_file.close()
                val_file.close()
                for ex in exma_list:
                    ex_csv_writer.writerow(ex)

                # print('loss',loss)
            coord.request_stop()
            coord.join(threads)


        val_writer.flush()
        train_writer.flush()
        val_writer.close()
        train_writer.close()


def main():
    try:
        train(learning_rate)
    except:
        sys.exit()

# log_dir='E:/Project/stacked_hourglass_net/log'
# if tf.gfile.Exists(log_dir):
#     tf.gfile.DeleteRecursively(log_dir)
# tf.gfile.MakeDirs(log_dir)

if __name__ == '__main__':
    main()
