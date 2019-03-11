import random
import os
import shutil


class Data_utils:
  """Class creates dataset required for training."""

  def __init__(self, input_csv_path, input_filepath, output_path, train_percentage,
    val_percentage, extension):
    self.traindata = 'train_dataset'
    self.valdata = 'val_dataset'
    self.testdata = 'test_dataset'
    self.input_csv_path = input_csv_path
    self.input_filepath = input_filepath
    self.output_path = output_path
    self.train_percentage = train_percentage
    self.val_percentage = val_percentage
    self.extension = extension

  def createDirectoryForDataset(self, dataname):
    """Function to create directiry for dataset."""
    try:
      os.makedirs(os.path.join(self.output_path, dataname))
    except Exception as e:
      pass

  def create_datasets(self):
    """Function to create datasets. """
    self.createdtraindata, self.createdval_data, self.createdtest_data = self.split_dataset()
    self.create_traindataset()
    self.create_validationdataset()
    self.create_testdataset()

  def split_dataset(self):
    """Function to split the data into training,validation and test set
    Returns:
       train_filenames,val_filenames and test_filenames as list"""
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
    """Function stores the split train data dataset.
    Returns:
     None """
    self.createDirectoryForDataset(self.traindata)
    for i in self.createdtraindata:
      newfilename = i[0]
      testfilename = newfilename + self.extension
      files = os.listdir(self.input_filepath)
      if testfilename in files:
        shutil.copy(self.input_filepath + testfilename, self.output_path + self.traindata)

  def create_validationdataset(self):
    """Function to store the split validation data set.
    Returns:
     None """
    self.createDirectoryForDataset(self.valdata)
    for i in self.createdval_data:
      newfilename = i[0]
      testfilename = newfilename + self.extension
      files = os.listdir(self.input_filepath)
      if testfilename in files:
        shutil.copy(self.input_filepath + testfilename, self.output_path + self.valdata)

  def create_testdataset(self):
    """Function stores the split test data set.
    Returns:
      None """
    self.createDirectoryForDataset(self.testdata)
    for i in self.createdtest_data:
      newfilename = i[0]
      testfilename = newfilename + self.extension
      files = os.listdir(self.input_filepath)
      if testfilename in files:
        shutil.copy(self.input_filepath + testfilename, self.output_path + self.testdata)
