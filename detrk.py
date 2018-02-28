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
                        default='model/labelmap_mot.prototxt')
    parser.add_argument('--model_def',
                        default='model/deploy.prototxt')
    parser.add_argument('--image_resize', default=512, type=int)
    parser.add_argument('--model_weights',
                        default='model/VGG_MOT_SSD_512x512_iter_40000.caffemodel')
    parser.add_argument('--det_conf_thresh', default=0.25, type=float)
    parser.add_argument('--seq_dir',default="sequence/")
    parser.add_argument('--sort_max_age',default=5,type=int)
    parser.add_argument('--sort_min_hit',default=3,type=int)
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    Detector=SSD(args.gpu_id,args.model_def, args.model_weights,args.image_resize, args.labelmap_file)
    mot_tracker = Sort(args.sort_max_age,args.sort_min_hit) 
    seqDir=args.seq_dir
    images=os.listdir(seqDir)
    images.sort(key=str.lower)
    colours = np.random.rand(32,3)*255
    for image_name in images:
        image_path=os.path.join(seqDir,image_name)
        result = Detector.detect(image_path,args.det_conf_thresh)
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
            cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(int(colours[label%32,0]),int(colours[label%32,1]),int(colours[label%32,2])),2)
            cv2.imshow("dst",im)
            cv2.waitKey(1)
        
      
    
    
