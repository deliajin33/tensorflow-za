#coding:utf-8
import tensorflow as tf
from PIL import Image, ImageFilter
from PIL import ImageOps
import os
import numpy as np
#import mnist_train
#import mnist_inference

PATH = '/Users/delia/Tests_Delia/A_Z/English/Hnd/Img'

def stronger(datas):
	for i in range(0,784):
		if(datas[0][i]>0 ):
			datas[0][i]=1
		else:
			datas[0][i]=0
	return datas

def pre_process(folderName):

	labels = np.zeros( (55,52) )
	images = np.zeros( (55,784) )
	
			
	label = folderName - 1
		
	img_dir = os.path.join(PATH, str(folderName))	

	count = 0

	for img in os.listdir(img_dir):		#进入遍历文件夹1,2,3...
		count = count + 1 

		image_file = Image.open(img_dir + '/' +  img)
		out = image_file.resize((28, 28)).convert('L')
		inverted_image = ImageOps.invert(out)
		data=np.array(inverted_image)
		datas=np.reshape(data,(1,-1))
		image = stronger(datas)
		images[count-1] = image
		labels[count-1][label] = 1
		#print images[count-1]
		#print labels[count-1][label]
		return images,labels		#每次返回一个文件夹下的图片和对应标签


def next_batch(folderName):
	images , labels = pre_process(folderName)
	return images,labels

if __name__ == '__main__':
	pre_process(12)
