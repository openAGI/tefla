class DataSet(object):
    def __init__(self, training_files, training_labels, validation_files, validation_labels):
        self.training_files = training_files
        self.training_labels = training_labels
        self.validation_files = validation_files
        self.validation_labels = validation_labels

    def print_info(self):
        print('Training data shape: %s; labels shape: %s' % (self.training_files.shape, self.training_labels.shape))
        print('Validation data shape: %s; labels shape: %s' %
              (self.validation_files.shape, self.validation_labels.shape))
