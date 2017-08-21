import numpy as np
from scipy import misc, ndimage
from skimage import filters


def flatten(image):
    """
    Flatten rgb image into 2-D grayscale image using luminosity method

    Args:
        image: rgb image array_like

    Returns:
         Flattened grayscale image

    """
    R, G, B = image.T
    flat = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return flat.T

def resize_proportions(img, size, dimension='width'):
    """
    Resize image retaining it's proportions

    :param img:     image as numpy array
    :param size:    image dimension
    :param orient:  dimension of size, default: width
    :return:        processed image
    """
    proportion = 1 - np.round((img.shape[1] - size)/img.shape[1], decimals=2)

    return misc.imresize(img, (int(img.shape[0] * proportion), size))


def extract_grade(image_name):
    """
    Select out grading strip and banknote from scan

    :param image_name:  path to image containing grading
    :return:            grading strip, banknote
    """

    """
    Read and preprocess image for fragmenting
    """
    org_image = misc.imread(image_name, mode='RGB')
    image = flatten(org_image)
    image = ndimage.gaussian_filter(image, sigma=9)

    """
    Apply otsu tresholding
    """
    labels = filters.threshold_otsu(image)
    mask = image < labels
    image[mask] = 0

    """
    Make image completly binary and fill holes created in the process
    """
    image[image != 0] = 1
    image = ndimage.binary_fill_holes(image)

    """
    Label objects in the image
    """
    image, count = ndimage.label(image)
    labels = ndimage.find_objects(image)

    """
    Weed out unwanted objects and select two widest as strip and banknote 
    """
    parts = np.array([image[label].shape[0] for label in labels])
    parts = parts.argsort()

    return org_image[labels[parts[-1]]], org_image[labels[parts[-2]]]

def process_grading(image_name):
    width = 890

    strip, note = extract_grade(image_name)
    strip, note = np.rot90(strip, k=-1), np.rot90(note, k=-1)

    return resize_proportions(strip, width), resize_proportions(note, width)

def glue_togather(image_name):
    strip_up, note_up = process_grading(image_name)

    padding_height = 10
    padding = np.zeros((padding_height, strip_up.shape[1], 3))
    up = np.concatenate((strip_up, padding, note_up))

    reverse_name = image_name.split('.')
    reverse_name = reverse_name[0] + '_r.' + reverse_name[1]

    strip_down, note_down = process_grading(reverse_name)

    padding = np.zeros((padding_height, strip_down.shape[1], 3))
    down = np.concatenate((note_down, padding, strip_down))

    result = np.concatenate((up, padding, down))

    return result




images_path = "images/"

result = glue_togather(images_path + '175_423.bmp')
misc.imsave('result.jpg', result)