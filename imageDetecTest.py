import numpy as np
import os
import time
#import matplotlib as plt
#%matplotlib inline
from PIL import Image
from PIL import ImageDraw

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/tianyi.cui/ssd/caffe/'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_cpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/tianyi.cui/ssd/caffe/data/mydataset/labelmap_mydataset.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_def = '/home/tianyi.cui/ssd/caffe/models/VGGNet/mydataset/SSD_300x300/deploy.prototxt'
model_weights = '/home/tianyi.cui/ssd/caffe/models/VGGNet/mydataset/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)


image_dir='/home/tianyi.cui/data/VOCdevkit/mydataset/JPEGImages/'
#image_dir='/home/tianyi.cui/ssd/image_from_internet/'
image_dir_list=os.listdir(image_dir)

timeList=[]

for each_image in image_dir_list:
#    startTime=time.clock()
    each_image_dir=os.path.join('%s%s' % (image_dir,each_image))
    image1=Image.open(each_image_dir)
    image = caffe.io.load_image(each_image_dir)
    #image1.save("/home/tianyi.cui/ssd/image1.jpg")
#    startTime=time.clock()
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    startTime=time.clock()
# Forward pass.
    detections = net.forward()['detection_out']

# Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

# Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    
    endTime=time.clock()
    detTime=endTime-startTime
    timeList.append(detTime)
#    print('Time for one image detection: %s seconds' %(detTime))

    colors = np.linspace(0, 0, 0).tolist()
#image2.save("/home/tianyi.cui/ssd/image2.jpg")

#currentAxis = plt.gca()
#draw=ImageDraw.Draw(image2)

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
   # color = colors[label]
   # imgtemp=Image.new("HSV",(1000,1000),(int(colors[label]),1,1))    
        draw=ImageDraw.Draw(image1)
        #print(coords)
        draw.rectangle((xmin,ymin,xmax,ymax))
        draw.text((xmin,ymin),display_txt)
    #currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
   # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    image1.save("/home/tianyi.cui/ssd/imageoutput/"+each_image+"_test.jpg")
timeListArray=np.array(timeList)
print('The average time to deal with one image is %s seconds' %(np.mean(timeListArray)))
