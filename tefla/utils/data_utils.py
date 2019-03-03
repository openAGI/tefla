import random
import os
from os import path
import shutil
import click
import csv


class Data_utils:
  """ Class creates dataset required for training"""

  def __init__(self, input_csv_path, input_filepath, output_path, train_percentage, val_percentage):
    self.traindata = 'train_dataset'
    self.valdata = 'val_dataset'
    self.testdata = 'test_dataset'
    self.input_csv_path = input_csv_path
    self.input_filepath = input_filepath
    self.output_path = output_path
    self.train_percentage = train_percentage
    self.val_percentage = val_percentage

  def createDirectoryForDataset(self, dataname):
    """
    createDirectoryForDataset function creates directory if it is not created by the user,
    for storing the dataset split for training,validation and test
    Args:
    path: path of the file that contains all the imagenames and groundtruth
    """
    try:
      os.makedirs(os.path.join(self.output_path, dataname))
    except:
      pass

  def create_datasets(self):
    """ create_datasets function call the functiosn that are used for creating the datasets """
    self.createdtraindata, self.createdval_data, self.createdtest_data = self.split_dataset()
    traindata = self.create_traindataset()
    valdata = self.create_validationdataset()
    testdata = self.create_testdataset()

  def split_dataset(self):
    """
    split_dataset function splits the data into training set, validation set and test set
    Args:
    imagelabelpath: path of the label, accepts csv file
    training_percent: percentage value to split the data for the training set
    val_percentage: percentage value to split the data for the validation set
    Returns:
    train_filenames,val_filenames and test_filenames as list
    """

    file = open(self.input_csv_path, "r")
    for _ in range(1):
      next(file)
      data = []
    for line in file:
      data.append(line.split(','))
    file.close()
    random.shuffle(data)
    n1 = self.train_percentage
    n2 = self.val_percentage
    traindata_split_1 = int(n1 * len(data))
    validationdata_split_2 = int(n2 * len(data))
    train_filenames = data[:traindata_split_1]
    val_filenames = data[traindata_split_1:validationdata_split_2]
    test_filenames = data[validationdata_split_2:]
    return train_filenames, val_filenames, test_filenames

  def create_traindataset(self):
    """ 
    create_traindataset function stores the data split by the createdata function in the location specified by the user
    Args:
    createdtraindata: traindata returned as list by createdata function
    original_path: path of the source directory from where the images are copied to train dataset
    train_datapath: path where the train dataset is to stored 
    """
    self.createDirectoryForDataset(self.traindata)
    for i in self.createdtraindata:
      newfilename = i[0]
      testfilename = newfilename + '.png'
      files = os.listdir(self.input_filepath)
      if testfilename in files:
        shutil.copy(self.input_filepath + testfilename, self.output_path + self.traindata)

  def create_validationdataset(self):
    """ 
    create_validationdataset function stores the split data in the location specified by the user
    Args:
    createdval_data: traindata returned as list by createdata function
    source_path:path of the source directory from where the images are copied to validation dataset
    valdatapath: path where the validation dataset is to stored
    """
    self.createDirectoryForDataset(self.valdata)
    for i in self.createdval_data:
      newfilename = i[0]
      testfilename = newfilename + '.png'
      files = os.listdir(self.input_filepath)
      if testfilename in files:
        shutil.copy(self.input_filepath + testfilename, self.output_path + self.valdata)

  def create_testdataset(self):
    """
    create_testdataset function stores the split data in the location specified by the user
    Args:
    createdtest_data: traindata returned as list by createdata function
    sourcepath: path of the source directory from where the images are copied to test dataset
    testdatapath: path where the test dataset is to stored
    """
    self.createDirectoryForDataset(self.testdata)
    for i in self.createdtest_data:
      newfilename = i[0]
      testfilename = newfilename + '.png'
      files = os.listdir(self.input_filepath)
      if testfilename in files:
        shutil.copy(self.input_filepath + testfilename, self.output_path + self.testdata)


# datacall = Data_utils("/home/raji/Desktop/test_2.csv",
#                       "/home/raji/Downloads/malaria/cell_images/test_p/",
#                       "/home/raji/Desktop/dataset/", 0.8, 0.9)

# print("the output is")
# datacall.create_datasets()
