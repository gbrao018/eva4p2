import numpy as np
import cv2
import io
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
import zipfile
from PIL import Image
import sys
import torch.utils.data
import timeit
import logging

# This is the common class for both test and train data sets. Based on indexes, this class serves the transformed image tensor.
  
#Considered 4 folders (n). Take all the files from those folders into a list, a global list in memory.     
#Flying birds, Large Quadcopters, Small QuadCopters, Winged drones
#Given an index, list[index] will be returned as a file.
#root_dir = "/content/drone-dataset"
class preprocess_images(Dataset):

  def __init__(self, root_dir, processed_root_dir, transform=ToTensor()):
    'Initialization'
    self.m_root = root_dir
    self.m_transform = transform
    logging.basicConfig(filename = "/content/prepreprocessing_timeit.log",datefmt='%H:%M:%S',level=logging.DEBUG)
    #self.g_dict_fileindex = dict()
    self.g_array_fileindex1 = []
    self.m_processed_root = processed_root_dir
    self.walkTheDirectory(self.m_root) 
    self.m_length = len(self.g_array_fileindex1)
  
  #root_dir structure
  #train
  #   /0
  #     file.jpg
  #     file40.jpg  
  #   /1
  #     filex.jpg
  #     filey.40.jpg  
  #   /2
  #     filea.jpg
  #     fileb.40.jpg  
  #   /3
  #     fileaa.jpg
  #     filebax.jpg  
  
  #test
  #   /0
  #     sfile.jpg
  #     mfile40.jpg  
  #   /1
  #     efilex.jpg
  #     cfiley.40.jpg  
  #   /2
  #     somefilea.jpg
  #     vfileb.40.jpg  
  #   /3
  #     somefileaa.jpg
  #     afilebax.jpg  
  
  
    
  def walkTheDirectory(self,root_dir):
    # for file in files:
    #    with open(os.path.join(root, file), "r") as auto: 
    subdirs = [x[0] for x in os.walk(root_dir)]
    dir = ''
    try:
      dir = self.m_processed_root
      os.mkdir(dir)
    except:
      print('dir present',self.m_processed_root)

    count = -1
    dir1 =  self.m_processed_root
    print('dir1=',dir1) 
    for d in subdirs:
      count = count + 1
      if count == 0:
        continue
      basedir = os.path.basename(subdirs[count])
      dir = os.path.join(dir1,basedir)
      print(dir)
      try:
        os.mkdir(dir)
      except:
        print('dir present', dir)
    
    for path, subdirs, files in os.walk(self.m_root): # path should give the complete path till the last directory.
        for filename in files:
            #print(os.path.join(path, filename))
            full_path = os.path.join(path, filename)
            #self.g_dict_fileindex[index] = full_path
            self.g_array_fileindex1.append(full_path)
            
 
  def __len__(self):
    'Denotes the total number of samples'
    return self.m_length;

  def printItem(self,idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
    file = self.g_array_fileindex1[idx]
    print('printFile-',file)  
  
  def removeItem(self,idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
    #self.g_array_fileindex.remove(file)

  def __getitem__(self, idx):
    if idx >= self.m_length:
        print('index', idx, 'is bigger than the actual dataset size')
        return
        
    start_time = timeit.default_timer()

    # Pick the file from the index
    size = 224, 224
    file = self.g_array_fileindex1[idx]
    
    file1 = file
    try:
      try:
        image = Image.open(file).convert('RGB') # even grey scale images will be converted to RGB with 3 channels
      except:
        raise "IOError"
      #temporarily we will resize to 224*224
      # We need to look into meaningful way of resizing (...pyramid pooling??)
      width,height = image.size
      #print('width=',width,'height=',height)
      image = image.resize((224,224), Image.BICUBIC)
      #print('image converted with BICUBIC')
      filename, file_extension = os.path.splitext(file)
      basedir = os.path.basename(os.path.dirname(filename)) # 0,1,2,3 dirs of either train or test
      #append subdir to new dir
      dir1 = os.path.join(self.m_processed_root,basedir) # This is new directory path.
    
      #Create jpg file
      basefile=os.path.basename(file)#This comes with extension
      #split with extension
      file1, ext = os.path.splitext(basefile)
      file1 = file1 + ".jpg"
      file1 = os.path.join(dir1,file1)#complete new file path
      #print('new filename=',file1)
      try:
        image.save(file1) # incase if it is already present
        #print('post proprocessed filename=',file1)
      except:
        print('post proprocessed filename=',file1)
        
      if self.m_transform:
        #print('transform to tensor')
        image = self.m_transform(image)
      
      self.g_array_fileindex1[idx] = file1
    except:
      print("exception while opening=",file)
      
    load_time = timeit.default_timer() - start_time
    
    desc = f' file={file1} orig_width={width},orig_height={height}, PROCESSING_TIME={load_time:0.3f}'
    logging.info(desc)
      
    return image
