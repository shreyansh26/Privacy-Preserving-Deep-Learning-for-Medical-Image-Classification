# -*- coding: utf-8 -*-

"""
Dataset Preprocessing
"""

from PIL import Image
import numpy as np
import pickle
import os

DATASET_PATH = './chest_xray/'
TRAIN_PATH_NORMAL = os.path.join(DATASET_PATH, 'train/', 'NORMAL/')
TRAIN_PATH_PNEUMONIA = os.path.join(DATASET_PATH, 'train/', 'PNEUMONIA/')
VALIDATION_PATH_NORMAL = os.path.join(DATASET_PATH, 'val/', 'NORMAL/')
VALIDATION_PATH_PNEUMONIA = os.path.join(DATASET_PATH, 'val/', 'PNEUMONIA/')
TEST_PATH_NORMAL = os.path.join(DATASET_PATH, 'test/', 'NORMAL/')
TEST_PATH_PNEUMONIA = os.path.join(DATASET_PATH, 'test/', 'PNEUMONIA/')

for file in os.listdir(TRAIN_PATH_NORMAL):
  print(file)
  if file == ".DS_Store":
    continue
  f = os.path.join(TRAIN_PATH_NORMAL, file)
  im = Image.open(f)
  a = np.asarray(im, dtype="int32")
  print(a)
  print(a.shape)
  newsize = (125, 150)
  im1 = im.resize(newsize)
  im1 = im1.convert('L')
  a = np.asarray(im1, dtype="int32")
  print(a)
  print(a.shape)
  break

print("**** TRAINING DATA - NORMAL ****")
normal_train = []

for file in os.listdir(TRAIN_PATH_NORMAL):
  print(file)
  if file == ".DS_Store":
    continue
  f = os.path.join(TRAIN_PATH_NORMAL, file)
  im = Image.open(f)
  newsize = (125, 150)
  im1 = im.resize(newsize)
  im1 = im1.convert('L')
  data = np.asarray(im1, dtype="int32")
  normal_train.append(data)

normal_train = np.asarray(normal_train)

with open("./normal_train.pkl", "wb") as f:
	pickle.dump(normal_train, f, protocol=4)

print("**** TRAINING DATA - PNEUMONIA ****")
pneumonia_train = []

for file in os.listdir(TRAIN_PATH_PNEUMONIA):
	print(file)
	if file == ".DS_Store":
		continue
	f = os.path.join(TRAIN_PATH_PNEUMONIA, file)
	im = Image.open(f)
	newsize = (125, 150)
	im1 = im.resize(newsize)
	im1 = im1.convert('L')
	data = np.asarray(im1, dtype="int32")
	pneumonia_train.append(data)

pneumonia_train = np.asarray(pneumonia_train)

with open("./pneumonia_train.pkl", "wb") as f:
	pickle.dump(pneumonia_train, f, protocol=4)

print("**** VALIDATION DATA - NORMAL ****")
normal_val = []

for file in os.listdir(VALIDATION_PATH_NORMAL):
	print(file)
	if file == ".DS_Store":
		continue
	f = os.path.join(VALIDATION_PATH_NORMAL, file)
	im = Image.open(f)
	newsize = (125, 150)
	im1 = im.resize(newsize)
	im1 = im1.convert('L')
	data = np.asarray(im1, dtype="int32")
	normal_val.append(data)

!ls '/content/drive/My Drive/BTP'

normal_val = np.asarray(normal_val)
with open("./normal_val.pkl", "wb") as f:
	pickle.dump(normal_val, f, protocol=4)

print("**** VALIDATION DATA - PNEUMONIA ****")
pneumonia_val = []

for file in os.listdir(VALIDATION_PATH_PNEUMONIA):
	print(file)
	if file == ".DS_Store":
		continue
	f = os.path.join(VALIDATION_PATH_PNEUMONIA, file)
	im = Image.open(f)
	newsize = (125, 150)
	im1 = im.resize(newsize)
	im1 = im1.convert('L')
	data = np.asarray(im1, dtype="int32")
	pneumonia_val.append(data)

pneumonia_val = np.asarray(pneumonia_val)
with open("./pneumonia_val.pkl", "wb") as f:
	pickle.dump(pneumonia_val, f, protocol=4)

print("**** TEST DATA - NORMAL ****")
normal_test = []

for file in os.listdir(TEST_PATH_NORMAL):
	print(file)
	if file == ".DS_Store":
		continue
	f = os.path.join(TEST_PATH_NORMAL, file)
	im = Image.open(f)
	newsize = (125, 150)
	im1 = im.resize(newsize)
	im1 = im1.convert('L')
	data = np.asarray(im1, dtype="int32")
	normal_test.append(data)

normal_test = np.asarray(normal_test)
with open("./normal_test.pkl", "wb") as f:
	pickle.dump(normal_test, f, protocol=4)

print("**** TEST DATA - PNEUMONIA ****")
pneumonia_test = []

for file in os.listdir(TEST_PATH_PNEUMONIA):
	print(file)
	if file == ".DS_Store":
		continue
	f = os.path.join(TEST_PATH_PNEUMONIA, file)
	im = Image.open(f)
	newsize = (125, 150)
	im1 = im.resize(newsize)
	im1 = im1.convert('L')
	data = np.asarray(im1, dtype="int32")
	pneumonia_test.append(data)

pneumonia_test = np.asarray(pneumonia_test)
with open("./pneumonia_test.pkl", "wb") as f:
	pickle.dump(pneumonia_test, f, protocol=4)

