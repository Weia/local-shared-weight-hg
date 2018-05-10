import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import  input_data

"""
残差网络
"""

nMoudel=1#hourglass 中residual 模块的数量
nStack=2#hourglass 堆叠的层数
nFeats=256 #hourglass 中特征图的数量
nPoint=16#关键点个数
def batch_norm(input_images):
    # Batch Normalization批归一化
    # ((x-mean)/var)*gamma+beta
    #输入通道维数
    #parms_shape=[input_images.get_shape()[-1]]
    #parms_shape=tf.shape(input_images)[-1]
    #print(parms_shape)
    #offset
    beta=tf.Variable(tf.constant(0.0,tf.float32),name='beta',dtype=tf.float32)
    #scale
    gamma=tf.Variable(tf.constant(1.0,tf.float32),name='gamma',dtype=tf.float32)
    #为每个通道计算均值标准差
    mean,variance=tf.nn.moments(input_images, [0, 1, 2], name='moments')
    y=tf.nn.batch_normalization(input_images,mean,variance,beta,gamma,0.001)
    y.set_shape(input_images.get_shape())

    return y


def batch_norm_relu(x):
    r_bn=batch_norm(x)
    r_bnr=tf.nn.relu(r_bn,name='relu')
    return  r_bnr


def conv2(input_images,filter_size,stride,in_filters,out_filters,padding='SAME',xvaier=True):

    n=filter_size*filter_size*out_filters
    #卷积核初始化
    if xvaier:

        weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weights')
    else:

        weights=tf.Variable(tf.random_normal(shape=[filter_size,filter_size,in_filters,out_filters],
                                         stddev=2.0/n,dtype=tf.float32),
                        dtype=tf.float32,
                        name='weights')
    biases=tf.Variable(tf.constant(0.0,shape=[out_filters]),dtype=tf.float32,name='biases')
    r_conv=tf.nn.conv2d(input_images,weights,strides=stride,padding=padding)
    r_biases=tf.add(r_conv,biases)
    return r_biases
def local_share_weight_conv2(input_images,filter_size,stride,out_filters,div_w=2,div_h=1,padding='SAME',weight=None,activate=tf.nn.relu):
    #将权重分模块做卷积
    in_filters=input_images.get_shape().as_list()[-1]
    #卷积核初始化

    part_results=[]
    num_part=div_w*div_h
    shape_input=input_images.get_shape().as_list()
    #每一部分的宽度和高度计算
    w_part_middle=shape_input[1]//div_w
    h_part_middle=shape_input[2]//div_h
    norm_part_width=shape_input[1]//div_w
    norm_part_height=shape_input[2]//div_h
    region_list=[]

    begin_w = 0
    begin_h = 0

    for part in range(num_part):
        change_line=False

        if (part+1)%div_w==0:
            part_width=shape_input[1]
            change_line=True
        else:
            part_width=w_part_middle+begin_w
        if num_part-(part+1)<div_w:
            part_height=shape_input[2]
        else:
            part_height=h_part_middle+begin_h
        region_list.append([begin_w,part_width,begin_h,part_height])

        begin_w = begin_w+norm_part_width if not change_line else 0
        begin_h = begin_h+norm_part_height if  change_line else begin_h
        #print(begin_w,begin_h)

    for i in range(num_part):
        # print(i)
        biase = tf.Variable(tf.constant(0.0, shape=[out_filters]), dtype=tf.float32, name='biases'+str(i))
        if weight:
            _weights=weight
        else:
            _weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weight1'+str(i))

        part_input = input_images[:, region_list[i][0]:region_list[i][1], region_list[i][2]:region_list[i][3], :]
        r_conv = tf.nn.conv2d(part_input, _weights, strides=stride, padding=padding)
        r_biases = tf.add(r_conv, biase)
        r_act = activate(r_biases)
        part_results.append(r_act)
    hori_result=[]
    for r_i in range(len(part_results)):
        if r_i%div_w==0:
            hori_result.append(tf.concat(part_results[r_i:r_i+div_w],axis=1))

    _result=tf.concat(hori_result,axis=2)
    # print(
    #       'result',_result.get_shape().as_list())
    relu_result=tf.nn.relu(_result,name='relu')
    return relu_result


def pad_conv2(input_x,pad,filter_size,stride,in_filters,out_filters,xvaier=True):
    #
    # input_image = tf.Variable([[[[11, 21, 31], [41, 51, 61]], [[12, 23, 32], [43, 53, 63]]],
    #                            [[[1, 2, 3], [4, 5, 6]], [[14, 24, 34], [45, 55, 65]]]])
    # padding = tf.Variable([[0, 0], [0, 0], [3, 3], [3, 3]])
    if xvaier:

        weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weights')
    else:
        weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
                                           dtype=tf.float32),
                          dtype=tf.float32,
                          name='weights')
    padded=tf.pad(input_x,paddings=pad)

    conv_valid=tf.nn.conv2d(padded,weights,stride,padding='VALID')
    return conv_valid


def bottleneck_residual(input_images,stride,in_filters,out_filters,filter_size=None):
    orig_x=input_images
    mid_channels=int(out_filters//2)#除法有些问题
    with tf.name_scope('r1'):
        with tf.name_scope('batch_norm_relu'):
            x=batch_norm_relu(input_images)
        with tf.name_scope('conv'):
            #input_images,filter_size,stride,in_filters,out_filters
            x=conv2(x,1,stride,in_filters,mid_channels)
    with tf.name_scope('r2'):
        with tf.name_scope('batch_norm_relu'):
            x=batch_norm_relu(x)
        with tf.name_scope('conv'):
            x=conv2(x,3,stride,mid_channels,mid_channels)
    with tf.name_scope('r3'):
        with tf.name_scope('batch_norm_relu'):
            x=batch_norm_relu(x)
        with tf.name_scope('conv'):
            x=conv2(x,1,stride,mid_channels,out_filters)
    with tf.name_scope('skip'):
        if in_filters==out_filters:
            with tf.name_scope('identity'):
                orig_x=tf.identity(orig_x)
        else:
            with tf.name_scope('conv'):
                orig_x=conv2(orig_x,1,stride,in_filters,out_filters)
    with tf.name_scope('sub_add'):
        # if in_filters!=out_filters:
        #     orig_x=conv2(orig_x,1,stride,in_filters,out_filters)
        result=tf.add(orig_x,x)
    return result


def down_sampling(x,ksize,strides,padding='VALID'):

    #下采样
    return tf.nn.max_pool(x,ksize,strides,padding=padding,name='max_pool')


def up_sampling(x):
    #反卷积实现
    # weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
    #                                        stddev=2.0 / n, dtype=tf.float32),
    #                       dtype=tf.float32,
    #                       name='weights')
    # tf.nn.conv2d_transpose(x,)
    #最近邻插值实现
    # new_width=x.shape[1]*2
    # new_height=x.shape[2]*2
    y=tf.image.resize_nearest_neighbor(x,tf.shape(x)[1:3]*2,name='upsampling')
    return y


def hourglass(input_x,output_filters,n):

    #n表示hourglass的阶数
    orig_x=input_x
    with tf.name_scope('conv_road'):
        with tf.name_scope('down_sampling'):
            x=down_sampling(input_x,[1,2,2,1],[1,2,2,1])
        with tf.name_scope('pre_residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual'+str(i+1)):
                    x=bottleneck_residual(x,[1,1,1,1],output_filters,output_filters)

        with tf.name_scope('hourglass'+str(n)):
            if n>1:
                x=hourglass(x,output_filters,n-1)
            else:
                x=bottleneck_residual(x,[1,1,1,1],output_filters,output_filters)
        with tf.name_scope('back_residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual'+str(i+1)):
                    x=bottleneck_residual(x,[1,1,1,1],output_filters,output_filters)
        with tf.name_scope('upsampling'):
            x=up_sampling(x)

    with tf.name_scope('skip_road'):
        with tf.name_scope('residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual'+str(i+1)):
                    orig_x=bottleneck_residual(orig_x,[1,1,1,1],output_filters,output_filters)
    with tf.name_scope('sub_add'):
        y=tf.add(x,orig_x)
    return y


def lin(input_x,in_filters,out_filters):
    #1*1卷积stride=1,卷积，bn，relu
    conv=conv2(input_x,1,[1,1,1,1],in_filters,out_filters)
    return batch_norm_relu(conv)


def model(input_x):
    #conv=conv2(input_x,7,[1,2,2,1])
    with tf.name_scope('conv_pad3'):
        cp=pad_conv2(input_x,[[0,0],[3,3],[3,3],[0,0]],7,[1,2,2,1],3,64)
    with tf.name_scope('batch_norm_relu'):
        bn=batch_norm_relu(cp)
    with tf.name_scope('residual1'):
        r1=bottleneck_residual(bn,[1,1,1,1],64,128)
    with tf.name_scope('down_sampling'):
        ds=down_sampling(r1,[1,2,2,1],[1,2,2,1])
    with tf.name_scope('residual2'):
        r2=bottleneck_residual(ds,[1,1,1,1],128,128)
    with tf.name_scope('residual3'):
        r3=bottleneck_residual(r2,[1,1,1,1],128,nFeats)
    h_input=r3#hourglass 的输入
    output=None
    for n in range(nStack):
        with tf.name_scope('hourglass'+str(n+1)):
            h1 = hourglass(h_input, nFeats, 4)
        residual=h1
        for i in range(nMoudel):
            with tf.name_scope('residual' + str(i + 1)):
                residual = bottleneck_residual(residual, [1, 1, 1, 1], nFeats, nFeats)
        with tf.name_scope('lin'):
            r_lin=lin(residual,nFeats,nFeats)
        with tf.name_scope('conv_same'):
            #局部共享权值
            output=local_share_weight_conv2(r_lin,1,[1,1,1,1],nPoint)
            #output=conv2(r_lin,1,[1,1,1,1],nFeats,nPoint,padding='VALID')#特征图输出
        if n<(nStack-1):
            #print(n)
            with tf.name_scope('next_input'):
                c_output=conv2(output,1,[1,1,1,1],nPoint,nFeats)#卷积的输出
                h_input=tf.add(h_input,tf.add(r_lin,c_output))
    #output=tf.reshape(output,(-1,16,64,64),name='output')
    output=tf.transpose(output,[0,3,1,2],name='output')#transpose和reshape结果是不一样的
    tf.summary.image('output',tf.transpose(output[0:1,:,:,:],[3,1,2,0]),max_outputs=16)
    return output


# def make_graph():
#     with tf.Graph().as_default():
#         with tf.name_scope('input'):
#             input_x=tf.placeholder(tf.float32,shape=[None, 256, 256, 3],name='input_x')
#             label=tf.placeholder(tf.float32,shape=[nStack,None,64,64,nPoint],name='label')
#         print('Input module ready')
#         with tf.name_scope('inference'):
#             train_model=model(input_x)
#         print('model module ready')
#
#         with tf.name_scope('loss'):
#             pass
#         with tf.name_scope('train'):
#             pass
#         with tf.name_scope('evaluation'):
#             pass
#         pass





# def main():
#     # log_dir='E:/Project/stacked_hourglass_net/log'
#     # if tf.gfile.Exists(log_dir):
#     #     tf.gfile.DeleteRecursively(log_dir)
#     # tf.gfile.MakeDirs(log_dir)
#
#     mnist = input_data.read_data_sets('./data')
#
#     input_image = mnist.train.next_batch(10)
#     with tf.Session() as sess:
#         input_image = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_images')
#         #result=bottleneck_residual(input_image,[1,1,1,1],128,128)
#         #result=hourglass(input_image,128,3)
#         model(input_image)
#         file_writer = tf.summary.FileWriter('log/', graph=tf.get_default_graph())
#         file_writer.flush()
#         file_writer.close()
#
#
#
#
#
#         #sess.run(model(input_image))
#
#
#
# if __name__ == '__main__':
#     #tf.nn.atrous_conv2d()
#     main()