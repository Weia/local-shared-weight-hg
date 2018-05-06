#生成train，test.txt文件
import numpy as np
import parse_annot
train_txt=[]#train image list
test_txt=[]#test images list
person_str=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R''S']
multiRef = np.ones(parse_annot.nimages)
trainRef = parse_annot.annot['img_train'][0][0][0]#is or not train images
allIdxs = np.arange(0,trainRef.shape[0])
#print(parse_annot.nimages,trainRef.shape[0])
imgnameRef = parse_annot.annot['annolist'][0][0][0]['image'][:]#names of images
#print(imgnameRef.shape)
# imgname = str(imgnameRef[1][0][0][0][0])+person_str[1]
# print(imgname)
for idx in range(parse_annot.nimages):#parse_annot.nimages
    #print(idx)

    for person in range(parse_annot.numpeople(idx)):
        print((idx,person))
        each_str = ''
        c, s = parse_annot.location(idx, person)
        if not c[0] == -1:
            imgname = str(imgnameRef[idx][0][0][0][0])+person_str[person]
            each_str+=imgname
            each_str+=' '
            print((imgname, person))
            if parse_annot.istrain(idx) == True:
                coords = np.zeros((16, 2))
                vis = np.zeros(16)
                for part in range(16):
                    coords[part], vis[part] = parse_annot.partinfo(idx, person, part)
                    for i in range(2):
                        each_str+=str(coords[part][i])
                        each_str += ' '
                train_txt.append(each_str)
            else:
                test_txt.append(each_str)
# print(len(train_txt),len(test_txt),len(train_txt+test_txt))
with open('train.txt','w') as f:
    for line in train_txt:
        f.write(line)
        f.write('\n')
f.close()
with open('test.txt','w') as f:
    for line in test_txt:
        f.write(line)
        f.write('\n')
f.close()





