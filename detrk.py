import os
import argparse
import cv2
import numpy as np
from ssd import SSD
from sort import Sort

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/home/cvpr/caffe/data/MOT/labelmap_mot.prototxt')
    parser.add_argument('--model_def',
                        default='/home/cvpr/caffe/models/MOT/SSD_512x512/deploy.prototxt')
    parser.add_argument('--image_resize', default=512, type=int)
    parser.add_argument('--model_weights',
                        default='/home/cvpr/caffe/models/MOT/SSD_512x512/'
                        'VGG_MOT_SSD_512x512_iter_40000.caffemodel')
    return parser.parse_args()

args=parse_args()
Detector=SSD(args.gpu_id,args.model_def, args.model_weights,args.image_resize, args.labelmap_file)
mot_tracker = Sort() 
seqDir="/home/cvpr/xcz/MOT/data/MOT16/train/MOT16-05/img1"
images=os.listdir(seqDir)
images.sort(key=str.lower)
colours = np.random.rand(32,3)*255
for image_name in images:
    image_path=os.path.join(seqDir,image_name)
    result = Detector.detect(image_path)
    im=cv2.imread(image_path)
    height=im.shape[0]
    width=im.shape[1]
    result=np.array(result)
    det=result[:,0:5]
    det[:,0]=det[:,0]*width
    det[:,1]=det[:,1]*height
    det[:,2]=det[:,2]*width
    det[:,3]=det[:,3]*height
    trackers = mot_tracker.update(det)
    for d in trackers:
        xmin=int(d[0])
        ymin=int(d[1])
        xmax=int(d[2])
        ymax=int(d[3])
        label=int(d[4])
        print label
        print colours[label%32,:]
        cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(int(colours[label%32,0]),int(colours[label%32,1]),int(colours[label%32,2])),1)
        cv2.imshow("dst",im)
        cv2.waitKey(1)
    
  


