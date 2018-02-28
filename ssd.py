"""
this is a class for SSD detector write by caffe python interface
a modefied version of ssd_detect.py in ssd_root/examples
"""
import os
import sys
import numpy as np
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

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

class SSD:
    def __init__(self,gpu_id,model_def,model_weights,image_resize,labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        
        self.image_resize=image_resize
        self.net=caffe.Net(model_def,model_weights,caffe.TEST)
        
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        
        self.transformer.set_raw_scale('data', 255)
        
        self.transformer.set_channel_swap('data', (2, 1, 0))
        
        file=open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)
    def detect(self, image_file, conf_thresh=0.25, topn=100):
   
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = float(top_xmin[i]) # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = float(top_ymin[i]) # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = float(top_xmax[i]) # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = float(top_ymax[i]) # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = float(top_conf[i])
#            label = int(top_label_indices[i])
#            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, score])
        return np.array(result)
    