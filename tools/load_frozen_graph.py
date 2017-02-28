# -------------------------------------------------------------------#
# Tool to load tenorflow frozen model def file for prediction
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#
import argparse
import tensorflow as tf
import cv2
import numpy as np


def load_frozen_graph(frozen_graph):
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    try:
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='model',
                op_dict=None,
                producer_op_list=None
            )
        return graph
    except Exception as e:
        print(e.message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--frozen_model",
        default="frozen_model.pb",
        type=str,
        help="Frozen model file to import")
    args, unparsed = parser.parse_known_args()

    graph = load_frozen_graph(args.frozen_model)

    for op in graph.get_operations():
        print(op.name)

    inputs = graph.get_tensor_by_name('model/inputs/input:0')
    predictions = graph.get_tensor_by_name('model/predictions/Softmax:0')

    im1 = cv2.imread(
        '/home/artelus_server/Downloads/test_512/Lim, Chu Luan_OD-ARMD.tiff', 1)
    im1 = im1[0:448, 0:448, :]
    im2 = cv2.imread(
        '/home/artelus_server/Downloads/test_512/Lim, Chu Luan_OD-ARMD.tiff', 1)
    im2 = im2[0:448, 0:448, :]
    im = []
    im.append(im1)
    im.append(im2)
    im = np.reshape(np.asarray(im), (2, 448, 448, 3))

    with graph.as_default():
        with tf.Session() as sess:
            pred = sess.run([predictions], {inputs: im})
            print(pred)
