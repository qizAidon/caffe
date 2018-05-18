import caffe
import numpy as np
import pdb
import cv2
import csv

def drawBox(img, xmin,ymin,xmax,ymax,conf,save_dir,coor_dir):
    """
    draw boxes given by two points.
    Note that coordinates of points have been normalized w.r.t. the width and height of original image to [0,1]
    """
#    pdb.set_trace()    
    
    gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h = img.shape[0]
    w = img.shape[1]
    xmin = max(int(np.round(xmin*w)),0)
    ymin = max(int(np.round(ymin*h)),0)
    xmax = min(int(np.round(xmax*w)),h)
    ymax = min(int(np.round(ymax*h)),w)
    
#    pdb.set_trace()
#    fin = open(coor_dir,'w')
    for i in range(len(xmin)):
#	fin.write(str(int(xmin[i]))+','+str(int(ymin[i]))+','+str(int(xmax[i]))+','+str(int(ymin[i]))+','+str(int(xmax[i]))+','+str(int(ymax[i]))+','+str(int(xmin[i]))+','+str(int(ymax[i]))+','+str(conf[i])+'\n')
#	fin.write(str(xmin[i])+','+str(ymin[i])+','+str(xmax[i])+','+str(ymax[i])+','+str(conf[i])+'\n')
        cv2.rectangle(img,(xmin[i],ymin[i]),(xmax[i],ymax[i]),(0,0,255),5)
	pdb.set_trace()
	box = gray_im[xmin[i]:xmax[i],ymin[i],ymax[i]]

#    fin.close()

    cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Result',500,500)
#    cv2.imwrite(save_dir,img)
    cv2.imshow('Result',box)
    cv2.waitKey(2000)

    cv2.destroyWindow('Result')
    pdb.set_trace()


##------------------------------------------------------------------
model_def = 'models/VGGNet/VOC0712/SSD_512x512/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_512x512/VGG_VOC0712_SSD_512x512_iter_40000.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.array([104,117,123]))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)

#------------------------------------------------------------------
#test_list = '/home/zq/add1/ssd/data/VOCdevkit/indoor/ImageSets/Main/test.txt'
#im_path = '/home/zq/add1/ssd/data/VOCdevkit/indoor/JPEGImages/'

test_list = '/home/zq/add1/ssd/data/test_name.txt'
im_path = '/home/zq/add1/ssd/data/icdar2017rctw_test/'
save_path = '/home/zq/add1/ssd/data/icdar2017rctw_res/'
coor_path = '/home/zq/add1/ssd/data/icdar2017rctw_coor/'
task1_path = '/home/zq/add1/ssd/data/icdar2017rctw_task1/'

#conf_th = 0.4
#conf_th = 0.3
conf_th = 0.2

with open(test_list) as inputfile:
    for row in csv.reader(inputfile):

#	pdb.set_trace()

	im_name = im_path + row[0]+'.jpg'
#	im_name = im_path + row[0]
#	im_name = '/home/zq/add1/ssd/data/VOCdevkit/indoor/JPEGImages/image_6758.jpg'
	image = caffe.io.load_image(im_name)
#	image = caffe.io.load_image('/home/zq/add1/ssd/data/VOCdevkit/indoor/JPEGImages/image_16.jpg')

	transformed_image = transformer.preprocess('data',image)
	net.blobs['data'].data[...] = transformed_image

	detections = net.forward()['detection_out']

	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]

	top_indices = [i for i, conf in enumerate(det_conf) if conf>=conf_th]

	top_conf = det_conf[top_indices]
	top_label_indices = det_label[top_indices].tolist()
	top_xmin = det_xmin[top_indices]
	top_ymin = det_ymin[top_indices]
	top_xmax = det_xmax[top_indices]
	top_ymax = det_ymax[top_indices]
	print(top_conf)

	save_dir = save_path + row[0] + '.jpg'
	coor_dir = coor_path + row[0] + '.txt'
	task1_dir = task1_path + row[0] + '.txt'
	print "Number of boxes = "+str(len(top_conf))
	if len(top_conf)>0:
#	     fin = open(coor_path+row[0]+'.txt','w')
             drawBox(image,top_xmin,top_ymin,top_xmax,top_ymax,top_conf,save_dir,task1_dir)

	#cv2.namedWindow('Test',cv2.WINDOW_NORMAL)
	#cv2.setWindowProperty('Test',500,500)
	#cv2.imshow('Test',image)
	#cv2.waitKey(0)
