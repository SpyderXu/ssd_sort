# SSD_SORT
This is a simple multi-person tracking system.<br>
* Detector:  [SSD](https://github.com/weiliu89/caffe/tree/ssd).<br>
* Association:  [Sort](https://github.com/abewley/sort).<br>

### Instruction

To run this code，you should install [SSD](https://github.com/weiliu89/caffe/tree/ssd) and build the python interface. A pretrained person detector caffemodel is also needed for the detector. Our pretrained [caffemodel](https://jbox.sjtu.edu.cn/l/cuaFUs) is also supplied. The image sequence should be stored under sequence folder. Some deploy files are under model folder. Run detrk.py to see the detection and association results. You can also edit some arguments to meet different situations.

### Result
<div align=center><img width="640" height="480" src="https://raw.githubusercontent.com/SpyderXu/ssd_sort/master/example.png"/></div>


### Citing


    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }
