import tensorflow as tf
from glob import glob
import re
import PIL.Image
import numpy as np
import cv2
import hashlib
import io
import csv

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_path', 'images', 'Path to images')
flags.DEFINE_string('masks_path', 'annotations/masks', 'Path to masks')
flags.DEFINE_string('classes_file', 'labels.txt', 'Path to label list')

FLAGS = flags.FLAGS
CLASS_IDS = None
CLASS_NAMES = None

def get_class_id(class_name):
    return CLASS_IDS[CLASS_NAMES.index(class_name)]

def get_mask_paths(filename):
    image_number = re.search('(\d*)\.jpg', filename).group(1)
    mask_files = FLAGS.masks_path + '/annotations/masks/*{:>04d}.png'.format(image_number)
    mask_paths = glob(mask_files)
    mask_paths = {re.search('_([a-zA-Z]*)\d', p).group(1): p for p in mask_paths}
    return mask_paths

def bounding_box(img):
    img = (img > 0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    xmin, xmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return ymin, ymax, xmin, xmax

def create_tf_example(example):

    mask_paths = get_mask_paths(example)

    with tf.gfile.GFile(example, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    key = hashlib.sha256(encoded_jpg).hexdigest()

    classes = []
    masks = []
    bboxes = []

    for label, mp in mask_paths.items():
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        ret, labels = cv2.connectedComponents(mask)

        # The first component is the background, so we skip it
        for l in range(1, ret):
            cardmask = np.zeros(labels.shape, dtype=np.uint8)
            cardmask[labels == l] = 1

            if np.sum(cardmask) > 2000:
                bbox = bounding_box(cardmask)
                classes.append(label)
                masks.append(cardmask)
                bboxes.append(bbox)
            else:
                print("Object ", label, "discarded, item too small:", np.sum(cardmask))

    #height = image.shape[1] # Image height
    #width = image.shape[0] # Image width
    width, height = image.size

    filename = example
    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = 'jpeg' # b'jpeg' or b'png'

    xmins = [bb[2] for bb in bboxes] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [bb[3] for bb in bboxes] # List of normalized right x coordinates in bounding box
                         # (1 per box)
    ymins = [bb[0] for bb in bboxes] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [bb[1] for bb in bboxes] # List of normalized bottom y coordinates in bounding box
                         # (1 per box)
    classes_text = classes # List of string class name of bounding box (1 per box)
    classes = list(map(get_class_id, classes)) # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    global CLASS_IDS
    global CLASS_NAMES
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    examples = glob('images/*.jpg')
    
    with open (FLAGS.classes_file, 'r') as f:
        CLASS_IDS = [int(row[0]) for row in csv.reader(f)]
        CLASS_NAMES = [row[1] for row in csv.reader(f)]

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    tf.app.run()