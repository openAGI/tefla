"""Resize and crop images to square, save as tiff."""
# -------------------------------------------------------------------#
# Tool to convert images
# Contact: mrinalhaloi11@gmail.com
# Copyright 2017, Mrinal Haloi
# -------------------------------------------------------------------#

import numpy as np
from PIL import Image, ImageFilter


def convert(fname, target_size=512):
  img = Image.open(fname).convert('RGB')

  blurred = img.filter(ImageFilter.BLUR)
  ba = np.array(blurred)
  h, w, _ = ba.shape

  if w > 1.2 * h:
    left_max = ba[:, :w // 32, :].max(axis=(0, 1)).astype(int)
    right_max = ba[:, -w // 32:, :].max(axis=(0, 1)).astype(int)
    max_bg = np.maximum(left_max, right_max)

    foreground = (ba > max_bg + 10).astype(np.uint8)
    bbox = Image.fromarray(foreground).getbbox()

    if bbox is None:
      print('bbox none for {} (???)'.format(fname))
    else:
      left, upper, right, lower = bbox
      # if we selected less than 80% of the original
      # height, just crop the square
      if right - left < 0.8 * h or lower - upper < 0.8 * h:
        print('bbox too small for {}'.format(fname))
        bbox = None
  else:
    bbox = None

  if bbox is None:
    bbox = square_bbox(img, fname)

  cropped = img.crop(bbox)
  resized = cropped.resize([target_size, target_size])
  return resized


def convert_gen(fname, target_size=512):
  try:
    img = Image.open(fname).convert('RGB')
    resized = img.resize([target_size, target_size])
    return resized
  except Exception:
    print('Corrupted Image file %s' % fname)


def full_bbox(img, fname):
  print("full bbox conversion done for image: %s" % fname)
  w, h = img.size
  left = 0
  upper = 0
  right = w
  lower = h
  return (left, upper, right, lower)


def square_bbox(img, fname):
  print("square bbox conversion done for image: %s" % fname)
  w, h = img.size
  left = max((w - h) // 2, 0)
  upper = 0
  right = min(w - (w - h) // 2, w)
  lower = h
  return (left, upper, right, lower)


def convert_square(fname, target_size):
  img = Image.open(fname)
  bbox = square_bbox(img)
  cropped = img.crop(bbox)
  resized = cropped.resize([target_size, target_size])
  return resized
