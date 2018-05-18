## Detecting Chinese Text in the wild

### Reference
    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

***

### Instructions
* We use the caffe branch from [SSD](https://github.com/intel/caffe/wiki/SSD:-Single-Shot-MultiBox-Detector)

* Datasets can be download from [RCTW](http://mclab.eic.hust.edu.cn/icdar2017chinese/dataset.html)

* related files <br />
	* ./data/indoor/create_data_indoor.sh
	* ./data/indoor/create_list_indoor.sh
	* ./data/labelmap_indoor.prototxt
	* ./scripts/detect_test.py
	* ./scripts/matching_test.py
	* ./python/selective_search_ijcv_with_python

### Results
![](https://github.com/qizAidon/text_detection/blob/master/res.png)

