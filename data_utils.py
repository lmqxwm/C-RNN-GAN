# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:58:27 2022

@author: Mengqi LIU
"""

import numpy as np
import os
import data_loading

def MinMaxScaler(data):
  """
  Min Max normalizer.
  """
  minn = np.min(data, 0)
  numerator = data - minn
  denominator = np.max(data, 0) - minn
  norm_data = numerator / (denominator + 1e-7)
  return norm_data, minn, denominator


class DataLoader(object):

  def __init__(self, datadir, select_validation_percentage, select_test_percentage, data_name, seq_len=24):
    self.datadir = datadir
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    self.seed_count = 0
    if not datadir is None:
      print ('Data loader: datadir: {}'.format(datadir))
      self.read_data(select_validation_percentage, select_test_percentage, data_name, seq_len)

  def read_data(self, select_validation_percentage, select_test_percentage, data_name, seq_len):
    """
    Load datasets with name "data_name". Decompose time-series data with sequence length of "seq_len" and mix them.
    Split all into train, validation and test set.
    
    return self.song as whole data sets.
    """
    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []
    self.songs['all'] = []
    
    if not data_name=="sine":
      current_file = os.path.join(self.datadir, data_name) + ".csv"

      ori_data = np.loadtxt(current_file, delimiter = ",", skiprows = 1)
      print ('From dataset {}'.format(current_file))
      self.num_features = ori_data.shape[-1]
        
      # Flip the data to make chronological data
      ori_data = ori_data[::-1]
      # Normalize the data
      ori_data, self.minn, self.deno = MinMaxScaler(ori_data)
    
      # Preprocess the dataset
      temp_data = []    
      # Cut data by sequence length
      for i in range(0, len(ori_data) - seq_len):
          _x = ori_data[i:i + seq_len]
          temp_data.append(_x)
      num_sample = len(ori_data) - seq_len
      
    else:
      temp_data = data_loading.sine_data_generation(10000, 24, 5)
      num_sample = 10000
      print ('From self-generated dataset sine')
      self.num_features = 5
      self.minn = 0
      self.deno = 1
    
        
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))    
    for i in range(len(temp_data)):
        self.songs['all'].append(temp_data[idx[i]])
    
    
    
    validation_len = 0
    test_len = 0
    if select_test_percentage:
      validation_len = int(float(select_validation_percentage/100.0)*len(num_sample))
      print (('validation len: {}'.format(validation_len)))
      self.songs['validation'] = self.songs['all'][0:validation_len]
    if select_test_percentage:
      test_len = int(float(select_test_percentage/100.0)*len(num_sample))
      print (('test len: {}'.format(test_len)))
      self.songs['test'] = self.songs['all'][validation_len:(validation_len+test_len)]
    self.songs['train'] = self.songs['all'][(validation_len+test_len):]
    
    return self.songs


  def get_batch(self, batchsize, part='train'):
    """
    return (batch_size, seq_len, d)
    """
    #print (('get_batch(): pointer: {}, len: {}, batchsize: {}'.format(self.pointer[part], len(self.songs[part]), batchsize))

    if self.songs[part]:
      num_ = len(self.songs[part])
      #np.random.seed(self.seed_count)
      ind_ = np.random.randint(num_-batchsize, size=1)[0]
      print("Seed is ", ind_)
      batch = self.songs[part][ind_:(ind_+batchsize)]
      return batch
    else:
      raise 'get_batch() called but self.songs is not initialized.'

  def get_seq_batch(self, batchsize, part='train'):
    """
    return (batch_size, seq_len, d)
    """
    #print (('get_batch(): pointer: {}, len: {}, batchsize: {}'.format(self.pointer[part], len(self.songs[part]), batchsize))

    if self.songs[part]:
      batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
      self.pointer[part] += batchsize
      return batch
    else:
      raise 'get_seq_batch() called but self.songs is not initialized.'
  
  def get_num_features(self):
    return self.num_features

  def get_minn(self):
    return self.minn

  def get_deno(self):
    return self.deno

  def get_all_oridata(self):
      return self.songs['all']