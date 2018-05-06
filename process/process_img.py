import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

WIDTH=256
HEIGHT=256

def bound_coord(label):
    """返回坐标中的最大值最小值"""

    re_label=label.reshape((-1,2))
    l_min=np.min(re_label,axis=0)
    #print(l_min)
    l_max=np.max(re_label,axis=0)
    #print(l_max)
    if l_min.all()==0:
        #print(l_min)
        new_label=np.copy(label)
        new_label[label==0]=1e4
        new_label=new_label.reshape((-1,2))
        l_min=np.min(new_label,axis=0)
        #print(l_min)
    #print(l_max)
    return l_min,l_max
            #print('00')

def crop_box(label,o_width,o_height,w_percent=0.15,h_percent=0.15):
    #padding=[[0,0],[0,0],[0,0]]
    """扩大percent后的坐标"""
    l_min,l_max=bound_coord(label)
    #print(l_min,l_max)
    width=l_max[0]-l_min[0]
    height=l_max[1]-l_min[1]
    #print(width,height)
    l_x_min=l_min[0]-width*w_percent
    l_x_max=l_max[0]+width*w_percent
    l_y_min = l_min[1] - height * h_percent
    l_y_max = l_max[1] + height * h_percent
    #print(l_y_min,l_y_max)
    if l_x_min<0:l_x_min=0
    if l_y_min<0:l_y_min=0
    if l_x_max>o_width-1:l_x_max=o_width
    if l_y_max>o_height-1:l_y_max=o_height
    #扩大后的长和宽
    # new_width=l_x_max-l_x_min
    # new_height=l_y_max-l_y_min
    # if new_height>new_width:
    #     gap=(new_height-new_width)//2
    #     bounds=((l_x_min-gap),(l_x_max+gap))
    #     if bounds[0]<0:
    #         padding[1][0]=int(abs(bounds[0]))
    #     if bounds[1]>o_width-1:
    #         padding[1][1]=int(bounds[1]-o_width)
    # elif new_height<new_width:
    #     gap=(new_width-new_height)//2
    #     bounds=((l_y_min-gap),(l_y_max+gap))
    #     if bounds[0]<0:
    #         padding[0][0]=int(abs(bounds[0]))
    #     if bounds[1]>o_height-1:
    #         padding[0][1]=int(bounds[1]-o_height)
    # l_x_min+=padding[1][0]
    # l_y_min+=padding[0][0]
    #
    # l_x_min=int(l_x_min)
    # l_x_max = int(l_x_max)
    # l_y_min = int(l_y_min)
    # l_y_max = int(l_y_max)
    #print(l_y_min, l_y_max)
    #print(o_height)
    width = l_x_max - l_x_min
    height = l_y_max - l_y_min
    gap = width - height

    if gap < 0:
        l_x_min -= int(abs(gap) // 2)
        l_x_max += int(abs(gap) // 2)
    elif gap > 0:
        l_y_min -= int(gap // 2)
        l_y_max += int(gap // 2)
    return (l_x_min,l_x_max,l_y_min,l_y_max)

def crop_label(label,x_min,y_min):


    r_label=label.reshape(-1,2)

    for i in range(r_label.shape[0]):
        if np.array_equal(r_label[i],[0,0]):
            #print(1)
            continue
        else:
            r_label[i,0]=r_label[i,0]-x_min#+padding[1][0]
            r_label[i,1]=r_label[i,1]-y_min#+padding[0][0]
    return r_label
def crop_image(img,l_x_min, l_x_max, l_y_min, l_y_max):
    #print('1',type(l_y_max))

    #print(type(l_x_max),type(l_x_min), type(l_y_min), type(l_y_max))
    #new_img=img[l_x_min:l_x_max,l_y_min:l_y_max]
    new_img = img.crop((l_x_min, l_y_min, l_x_max, l_y_max))  # left, upper, right, and lower
    return new_img
def crop_data(img,label):

    #img=Image.open(os.path.join(img_dir,imgName))
    x,y=img.size
    #print(x,y)
    # fig1 = plt.figure()
    # ax2 = fig1.add_subplot(1, 2, 1)
    # ax1=fig1.add_subplot(1,2,2)
    # ax2.axis([0,x,y,0],'normal')
    # ax2.imshow(img)
    # o_label=label.reshape(-1,2)
    # o_x=o_label[:,0]
    # o_y=o_label[:,1]
    # ax2.plot(o_x, o_y, 'g*')
    l_x_min, l_x_max, l_y_min, l_y_max = crop_box(label, x, y)
    #print(l_x_min, l_x_max, l_y_min, l_y_max )
    #padded_img=np.pad(img,padding,'constant')
    #padded_img=Image.fromarray(padded_img)
    new_img=crop_image(img,l_x_min, l_x_max, l_y_min, l_y_max )

    #new_x,new_y=new_img.size
    c_label=crop_label(label,l_x_min,l_y_min)#(16,2)
    return new_img,c_label

    # ax1.axis([0,new_x,new_y,0])
    # x=c_label[:,0]
    # y=c_label[:,1]
    # ax1.plot(x,y,'r*')
    # ax1.imshow(new_img)
    #print(c_label)
    #print(l_x_min, l_x_max, l_y_min, l_y_max )

    #rect=plt.Rectangle((l_x_min,l_y_min),l_x_max-l_x_min,l_y_max-l_y_min,alpha=0.9)
    #ax1.add_patch(rect)
    #plt.show()

def resize_image(img,label):

    o_x, o_y = img.size  # 图像原始大小
    re_img = img.resize((WIDTH, HEIGHT),Image.ANTIALIAS)
    re_x, re_y = re_img.size  # 图像resize后的大小
    label[:,0]=label[:,0]*re_x/o_x

    label[:,1] = label[:,1] *re_y/o_y
    # for i in range(len(label)):
    #     if i%2==0:
    #         l=label[i]*re_x/o_x
    #         x.append(l)#后
    #         #ox.append(label[i])
    #     else:
    #         l=label[i]*re_y/o_y
    #         y.append(l)#后
    #         #oy.append(label[i])
    #     re_label.append(l)
    return re_img,label
#
# img_dir='E:\数据集\MPII\mpii_human_pose_v1\images'
#
# f=open('../train1.txt')
# lines=f.readlines()
# for line in lines:
#     line1=line.split(' ')
#     imgName=line1[0][:-1]
#     img=Image.open(os.path.join(img_dir,imgName))
#     label=line1[1:-1]
#     label=np.array(list(map(float,label)))
#     image,clabel=crop_data(img,label)
#     print(clabel)
#     #clabel=crop_label(label,-100,-100)
#     #image=image.resize((256,256))
#     #print(clabel)
#     re_iamge,relabel=resize_image(image,clabel)
#     #print(relabel)
#     x, y = re_iamge.size
#     plt.axis([0,x,y,0])
#     plt.imshow(re_iamge)
#     #plt.plot(clabel[:,0],clabel[:,1],'r*')
#     plt.plot(relabel[:, 0], relabel[:, 1], 'r*')
#     plt.show()


