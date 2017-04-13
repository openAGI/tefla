import click
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.cluster import KMeans
from scipy import ndimage
import matplotlib.pyplot as plt
from tefla.convert import convert


def process_image(image_filename, image_size):
    image = convert(image_filename, image_size)
    image = img_as_float(image)
    return image


def segment_image(image, numSegments, numClusters):
    segments = slic(image, n_segments=numSegments,
                    compactness=1.5, max_iter=50, sigma=8, convert2lab=True)
    idxs = []
    means = []
    stds = []
    maxrgb = []
    minrgb = []
    for i in range(numSegments):
        idxs.append(np.where(segments == i))
        means.append(np.mean(image[idxs[i][0], idxs[i][1], :], axis=(0)))
        stds.append(np.std(image[idxs[i][0], idxs[i][1], :], axis=(0)))
        try:
            maxrgb.append(np.max(image[idxs[i][0], idxs[i][1], :], axis=(0)))
            minrgb.append(np.min(image[idxs[i][0], idxs[i][1], :], axis=(0)))
        except Exception:
            maxrgb.append((0, 0, 0))
            minrgb.append((0, 0, 0))
    means = np.reshape(np.asarray(means, dtype=np.float32), (numSegments, 3))
    stds = np.reshape(np.asarray(stds, dtype=np.float32), (numSegments, 3))
    maxrgb = np.reshape(np.asarray(maxrgb, dtype=np.float32), (numSegments, 3))
    minrgb = np.reshape(np.asarray(minrgb, dtype=np.float32), (numSegments, 3))
    features = np.concatenate((means, stds), axis=1)
    nanidx = np.argwhere(np.isnan(features))
    features[nanidx] = 0.0
    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(features)

    for i in range(numSegments):
        segments[idxs[i][0], idxs[i][1]] = kmeans.labels_[i]

    all_labeled = []
    for i in range(numClusters):
        labeled, nr_objects = ndimage.label(segments == i)
        for j in range(nr_objects):
            idx = np.where(labeled == j)
            if len(idx[0]) > 500:
                labeled[idx] = 0
        class_idx = np.where(labeled > 0)
        if len(class_idx[0]) > 4000:
            labeled[class_idx] = 0
        else:
            labeled[class_idx] = i
        all_labeled.append(labeled)

    segment = all_labeled[0].copy()

    for i in range(1, numClusters):
        segment = np.add(segment, all_labeled[i])
    plt.imshow(segment)

    fig = plt.figure("segments")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segment))
    plt.axis("off")
    plt.show()


@click.command()
@click.option('--image_filename', show_default=True,
              help="path to image.")
@click.option('--num_segments', default=2000, show_default=True,
              help="Number of segmented region for slic")
@click.option('--num_clusters', default=30, show_default=True,
              help="Num clusters")
@click.option('--image_size', default=896, show_default=True,
              help="Size of converted images.")
def main(image_filename, num_segments, num_clusters, image_size):
    image = process_image(image_filename, image_size)
    segment_image(image, num_segments, num_clusters)


if __name__ == '__main__':
    main()
