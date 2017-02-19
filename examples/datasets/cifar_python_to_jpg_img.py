import os
import numpy as np
import sys
import cv2
import pickle


def cifar_pkl_to_img(pkl_dir, img_dir, label_filename):
    files = os.listdir(pkl_dir)
    fl = open(label_filename, 'w')
    for pfile in files:
        with open(os.path.join(pkl_dir, pfile), 'r') as f:
            data = pickle.load(f)
        images = data['data']
        num_images = images.shape[0]
        images = images.reshape((num_images, 3, 32, 32))
        labels = data['labels']
        filenames = data['filenames']

        for i in range(0, images.shape[0]):
            image = np.squeeze(images[i]).transpose((1, 2, 0))
            cv2.imwrite(os.path.join(img_dir, filenames[i]), image)
            fl.write(os.path.abspath(os.path.join(img_dir, filenames[
                     i])) + ' ' + str(labels[i]) + '\n')


if __name__ == '__main__':
    cifar_pkl_to_img(sys.argv[1], sys.argv[2], sys.argv[3])
