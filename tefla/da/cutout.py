import numpy as np


class Cutout(object):
  """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

  def __init__(self, n_holes, length):
    self.n_holes = n_holes
    self.length = length

  def __call__(self, img):
    """
        Args:
            img: `np.ndarrray` image of size (C, H, W). Channel First

        Returns:
            `np.ndarray`: Image with n_holes of dimension length x length cut out of it.
        """
    h = img.shape[1]
    w = img.shape[2]

    mask = np.ones((h, w), np.float32)

    for n in range(self.n_holes):
      y = np.random.randint(h)
      x = np.random.randint(w)

      y1 = int(np.clip(y - self.length / 2, 0, h))
      y2 = int(np.clip(y + self.length / 2, 0, h))
      x1 = int(np.clip(x - self.length / 2, 0, w))
      x2 = int(np.clip(x + self.length / 2, 0, w))

      mask[y1:y2, x1:x2] = 0.

    mask = np.tile(np.expand_dims(mask, axis=0), (3, 1, 1))
    img = img * mask
    return img
