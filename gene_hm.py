import numpy as np
import tensorflow as tf
HM_HEIGHT=64
HM_WIDTH=64
nPoints=16
def resize_label(label):
    return label*64/256
def _makeGaussian(height, width, sigma, center):
    """
    以center为中心生成值逐渐减小的矩阵，中心值为一
    :param height:
    :param width:
    :param sigma:
    :param center:
    :return: 一个height和width的矩阵
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2.0) * ((x-x0)**2 + (y-y0)**2) / sigma**2)
    #return np.exp(-4 * np.log(2.0) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)



def generate_hm(height, width ,joints, maxlenght):
    num_joints = joints.shape[0]
    r_hm = np.zeros((num_joints,height, width), dtype=np.float32)
    for i in range(num_joints):
        if np.array_equal(joints[i],[0,0]):
            r_hm[i] = np.zeros((height, width), dtype=np.float32)
        else:
            # if not (np.array_equal(joints[i], [0, 0])) :#and weight[i] == 1:
            s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
            r_hm[i]= _makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))

    return r_hm
def batch_genehm(batch_size,l):
    label = np.zeros((batch_size, nPoints, HM_HEIGHT,HM_WIDTH), dtype=np.float32)
    for i in range(batch_size):
        label[i] =generate_hm(HM_HEIGHT,HM_WIDTH, l[i], 64)
    return label


# joints=np.array([[1,1],[15.1,15.1]])
# weight=[1,1]
# hm=generate_hm(64,64,joints,2)
# print(hm.shape)
# print(hm[1,14:16,14:16])