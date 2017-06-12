'''
Face rec using pretrained tensorflow resnet
'''

import numpy as np
import cv2
import copy
import time
import json
import sys
import ast
import os
from os import listdir
from os.path import isfile, join
import math
import dlib
from PIL import Image as Image
import cnn_resnet_v1 as resnet
#import face_trainer as trainer
import tensorflow as tf
from tensorflow.python.platform import gfile
#crop face


#pretrained models from dlib
predictor_path = "shape_outliner.dat"
face_detect = dlib.get_frontal_face_detector();
face_marker = dlib.shape_predictor(predictor_path);

class FACE(object):
	def __init__(self, region, name, probability, eye_angle,face_pos):
		self.region = region;
		self.name = name;
		self.probability = probability;
		self.eye_angle = eye_angle;
		self.face_pos = face_pos;

def preprocessData():
	dataPath = "processing"
	onlyfiles = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
	index = 0;
	for image in onlyfiles:
		black_white = cv2.imread("processing/"+image,cv2.IMREAD_GRAYSCALE);
		faces = face_detect.detectMultiScale(black_white,1.3,1);
		if len(faces) > 0: 
			for (x,y,w,h) in faces: # x - y width height
				face = cropFace2(x,y,w,h,black_white);
				cv2.imwrite( "images/person"+str(index)+".jpg", face);
				index+=1;


#Tensorflow settings
x1 = tf.placeholder('float', [None, 160,160,3]) 
print("Loading Graph")
get_128D =  tf.nn.l2_normalize(resnet.inference(x1,0.6,phase_train=False)[0],1, 1e-10);
saver = tf.train.Saver();
print("Graph loaded")




def cameraDetection():
	input_stream = cv2.VideoCapture(0); #camera
	with tf.Session() as session:
		#session.run(tf.global_variables_initializer());
		saver.restore(session, 'trained/model-20170512-110547.ckpt-250000')
		#print(tf.all_variables())
		frames = 0;
		faces = [];
		features_128D = None;
		scale_factor = 4
		face_info = []
		while True: 
			ret, img = input_stream.read(); #capture
			img_resized = cv2.resize(img,(int(len(img[0])/scale_factor ),int(len(img)/scale_factor))); # resize for quicker face detection
			black_white = cv2.equalizeHist(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY));
			if(frames % 2 == 0): #skip frames
				face_info = [];
				faces = face_detect(black_white,1);
				if len(faces) > 0:
					for region in faces: # x - y width heighh
						landmarks = face_marker(black_white,region);
						face_pos = getFacePosition(landmarks);
						real_region = dlib.rectangle(max(region.left() * scale_factor-32,0),max(region.top() * scale_factor-32,0),min(region.right() * scale_factor+32,len(img[1])),min(region.bottom() * scale_factor+32,len(img)));
						crop_phase0 = prewhiten(cropFace(img,landmarks,scale_factor)); #preprocess step 1
						aligned_face, angle = alignFace(landmarks,crop_phase0) #align face using eye angle
						aligned_face = cv2.resize(aligned_face,(180,180));
						aligned_face = aligned_face[10:len(aligned_face)-10,10:len(aligned_face[1])-10]; #margin 32 to remove black bars after rotating
						cv2.imshow("aligned",aligned_face)
						features_128D = np.asarray(session.run(get_128D, feed_dict={x1 : [aligned_face]}));	
						person_name, prob = findPerson(features_128D,face_pos)[0]
						face_info.append(FACE(real_region,person_name,prob,angle,face_pos))

			for info in face_info:
				cv2.rectangle(img,(info.region.left(),info.region.top()),(info.region.right(),info.region.bottom()),(255,0,0),3);
				cv2.putText(img,"Face_pos:"+info.face_pos+". Eye Angle: "+str(angle),(info.region.left(),info.region.top()-25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
				cv2.putText(img,str(info.name)+". "+str(info.probability)+"%.",(info.region.left(),info.region.top()),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

			cv2.imshow("facial recognition",img)
			frames+=1;
			cv2.waitKey(1)
	input_stream.release();


def alignFace(landmarks,face):
	w,h = len(face[0]),len(face)
	eye1_x, eye1_y = landmarks.parts()[39].x,landmarks.parts()[39].y
	eye2_x, eye2_y = landmarks.parts()[42].x,landmarks.parts()[42].y
	angle = np.arctan((eye1_y - eye2_y) / (eye1_x - eye2_x)) * 180 / math.pi;
	M = cv2.getRotationMatrix2D((int(w/2),int(h/2)),angle,1)
	face = cv2.warpAffine(face,M,(w,h))	
	return face,angle

def cropFace(aligned_face,landmarks,scale_factor): #regular rectangular crop -- minimize background area
	points = landmarks.parts()
	min_x, min_y = sys.maxsize, sys.maxsize;
	max_x,max_y = 0, 0;
	for point in points:
		if(point.x < min_x):
			min_x = point.x
		if(point.x > max_x):
			max_x = point.x
		if(point.y < min_y):
			min_y = point.y;
		if(point.y > max_y):
			max_y = point.y;
	if(max_x != 0 and max_y != 0):
		return aligned_face[max(0,min_y * scale_factor):max_y*scale_factor,max(0,min_x*scale_factor):max_x*scale_factor]
	return aligned_face;

def cropFace2(img_face): #ellipse crop
	w,h = len(img_face[0]),len(img_face)
	face_center = (int(w/2), int(h*0.5)); #the center of 2 eyes

	mask = np.zeros((h,w,3), np.uint8)
	cv2.ellipse(mask, face_center, (int(w*0.5), int(h*0.5)), 0, 0, 360, (255,255,255), -1);
	for y in range(len(mask)):
		for x in range(len(mask[0])):
			if mask[y][x].tolist() != [255,255,255]:
				img_face[y][x] = [0,0,0];
	return img_face

#Facenet paper
def getFacePosition(landmarks):
	markers = landmarks.parts();
	if(markers[28].x - markers[0].x < (markers[16].x - markers[0].x) / 2):
		return "Left";
	elif((markers[1].x - markers[0].x) / 2 > (markers[2].x - markers[1].x)):
		return "Right";
	else:
		return "Center";

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def findPerson(features_arr, position, thres = 0.5):
	f = open('./facenet_128D.txt','r')
	data_set = json.loads(f.read());
	returnRes = [];
	for features_128D in features_arr:	
		result = "Unknown";
		smallest = sys.maxsize
		for person in data_set.keys():
			person_data = data_set[person][position];
			avg_distance = 0;
			for data in person_data:
				distance = np.sqrt(np.sum(np.square(data-features_128D)))#;
				#distance = np.linalg.norm(data-features_128D);
				#avg_distance += distance;
				#avg_distance = avg_distance / len(person_data);
				if(distance < smallest):
					smallest = distance;
					# percentage =  min(100, 100 * thres / smallest);
					# if(percentage > 70):
					result = person;
		percentage =  min(100, 100 * thres / smallest);
		returnRes.append((result,percentage))
	return returnRes;


def create_manual_data():
	input_stream = cv2.VideoCapture(0); #camera
	f = open('./facenet_128D.txt','r')
	data_set = json.loads(f.read());
	print("Input ID của người cần nhận dạng: ");
	a = input();
	start = time.time() #last pic
	scale_factor = 4
	if(data_set.get(a) == None):
		data_set[a] = {"Left" : [], "Right": [], "Center": []};
	position = {"Left": [], "Right": [], "Center":[]};
	with tf.Session() as session:
		saver.restore(session, 'trained/model-20170512-110547.ckpt-250000')
		while True: 
			ret, img = input_stream.read(); #capture
			img_resized = cv2.resize(img,(int(len(img[0])/scale_factor ),int(len(img)/scale_factor)));
			black_white = cv2.equalizeHist(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY));
			faces = face_detect(black_white,1); #can use dlib or opencv 
			if len(faces) == 1: 
				region = faces[0]
				landmarks = face_marker(black_white,region);
				face_pos = getFacePosition(landmarks);
				crop_phase0 = prewhiten(cropFace(img,landmarks,scale_factor));
				aligned_face, angle = alignFace(landmarks,crop_phase0)
				aligned_face = cv2.resize(aligned_face,(192,192));
				aligned_face = aligned_face[16:len(aligned_face)-16,16:len(aligned_face[1])-16]; #margin 32 to remove black bars after rotating
				cv2.imshow("Preprocessed", aligned_face);
				features_128D = np.asarray(session.run(get_128D, feed_dict={x1 : [aligned_face]})).tolist();
				position[face_pos].append(features_128D[0]);
				#a = input();
			if cv2.waitKey(33) == ord('a'):
				for item in position.keys():
					position[item] = [np.mean(np.asarray(position[item]),axis=0).tolist()];

				data_set[a] = position;
				f = open('./facenet_128D.txt','w');
				f.write(json.dumps(data_set));
				print("File saved: "+a);
	input_stream.release();


def test_data_set():
	dataPath = "train_data_set"
	while True:
		face = cv2.imread(str(dataPath)+"/12_1495912848.734331.jpg",cv2.IMREAD_GRAYSCALE) # get image from dataset
		face = cv2.resize(face, (48,48));
		cv2.rectangle(face,(24,0),(48,48),0,-1)
		feed = preprocessFace(face);
		cv2.imshow('LocalBinaryHists', cv2.resize(feed / 255., (280, 280)))
		with tf.Session() as session:
		 	saver.restore(session, "trained/facialrec.ckpt");
		 	print("Person id: "+str(session.run(result, feed_dict={x1: [feed]})));
		cv2.waitKey(10)
	cv2.destroyAllWindows();
#test_data_set(); # to test trained data set
#preprocessData();
cameraDetection(); # test trained data set with live camera 
#create_manual_data(); # to create manual data
