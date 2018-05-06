from PIL import Image
import os

images_path='/home/weic/project/linux/image'
names=os.listdir(images_path)
for name in names:

    image_path=os.path.join(images_path,name)
    or_image=Image.open(image_path)

    mirror_image=or_image.transpose(Image.FLIP_LEFT_RIGHT)
    split_name=name.split('.')[0]
    print(type(split_name))
    mirror_image.save('mirror'+split_name+'.jpg')