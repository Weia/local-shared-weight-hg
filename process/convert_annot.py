#将annotation内容存入字典
import numpy as np
#import scipy
import sys
import parse_annot
import pickle
keys=['index','person','imgname','center','scale','part','visible','normalize','torsoangle','multi','istrain']
train_annot = {k:[] for k in keys}
test_annot={k:[] for k in keys}
annot={k:[] for k in keys}
multiRef = np.ones(parse_annot.nimages)
trainRef = parse_annot.annot['img_train'][0][0][0]
allIdxs = np.arange(0,trainRef.shape[0])
#print(parse_annot.nimages,trainRef.shape[0])
imgnameRef = parse_annot.annot['annolist'][0][0][0]['image'][:]
# print(type(imgnameRef))
# imgname = np.zeros(16)
# refname = str(imgnameRef[1][0][0][0][0])
# print(refname)
# for i in range(len(refname)): imgname[i] = ord(refname[i])
# print(imgname)
def feed_annot_dict(isTrain,annot,idx,person):
    annot['index'] += [idx]
    annot['person'] += [person]
    # imgname = np.zeros(16)
    imgname = str(imgnameRef[idx][0][0][0][0])
    # for i in range(len(refname)): imgname[i] = ord(refname[i])#返回unicode编码
    annot['imgname'] += [imgname]
    annot['center'] += [c]
    annot['scale'] += [s]
    annot['multi'] += [multiRef[idx]]
    if isTrain==1:
        coords = np.zeros((16, 2))
        vis = np.zeros(16)
        for part in range(16):
            coords[part], vis[part] = parse_annot.partinfo(idx, person, part)
        annot['part'] += [coords]
        annot['visible'] += [vis]
        annot['normalize'] += [parse_annot.normalization(idx, person)]
        annot['torsoangle'] += [parse_annot.torsoangle(idx, person)]
        annot['istrain'] += [1]
    else:
        annot['part'] += [-np.ones((16, 2))]
        annot['visible'] += [np.zeros(16)]
        annot['normalize'] += [1]
        annot['torsoangle'] += [0]
        if trainRef[idx] == 0:  # Test image
            annot['istrain'] += [0]
        else:  # Training image (something missing in annot)
            annot['istrain'] += [2]
for idx in range(10):#parse_annot.nimages
    print(idx)
    for person in range(parse_annot.numpeople(idx)):
        c,s=parse_annot.location(idx,person)
        if not c[0]==-1:
            if trainRef[idx]==1:
                feed_annot_dict(1,train_annot,idx,person)
            elif trainRef[idx]==0:
                feed_annot_dict(0, test_annot, idx, person)

f=open('train_annot.txt','wb')
pickle.dump(train_annot,f)
f.close()
f=open('test_annot.txt','wb')
pickle.dump(test_annot,f)
f.close()
# print(train_annot['part'][0])
# print(test_annot)
# print(train_annot['index'],'/',test_annot['index'])

