class DataSet(object):
    def __init__(self, training_X, training_y, validation_X, validation_y):
        self.training_X = training_X
        self.training_y = training_y
        self.validation_X = validation_X
        self.validation_y = validation_y

    def print_info(self):
        print('Training data shape: %s; labels shape: %s' % (self.training_X.shape, self.training_y.shape))
        print('Validation data shape: %s; labels shape: %s' %
              (self.validation_X.shape, self.validation_y.shape))
