import glob

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

def adjust_contrast(image, C):
    """
    Adjust RGB image's contrast to desired level

    :param image: image to process
    :param C: contrast level
    :return: adjusted image
    """
    F = (259 * (C + 255)) / (255 * (259 - C))

    def contrast(Colour):
        return F * (Colour - 128) + 128

    def trunctuate(colour):
        colour[colour < 0] = 0
        colour[colour > 255] = 255

        return colour

    R, G, B = image.T.astype(np.float16)

    image.T[0], image.T[1], image.T[2] = \
        trunctuate(contrast(R)), trunctuate(contrast(G)), trunctuate(contrast(B))

    return image


def resize_proportions(img, size, dimension='width'):
    """
    Resize image retaining it's proportions

    :param dimension:
    :param img:     image as numpy array
    :param size:    image dimension
    :param dimension: dimension of size, default: width
    :return:        processed image
    """
    proportion = 1 - np.round((img.shape[1] - size) / img.shape[1], decimals=2)

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
    image = ndimage.gaussian_filter(image, sigma=30)

    """
    Apply otsu thresholding
    """
    labels = filters.threshold_otsu(image)
    mask = image < labels
    image[mask] = 0

    """
    Make image completely binary and fill holes created in the process
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


def process_scan(image_name):
    """
    image processing, resize and aesthetics

    :param image_name: name of image to process
    :return:           processed strip, processed banknote
    """
    width = 890

    strip, note = extract_grade(image_name)
    strip, note = np.rot90(strip, k=-1), np.rot90(note, k=-1)

    strip, note = adjust_contrast(strip, 64), adjust_contrast(note, 64)

    return resize_proportions(strip, width), resize_proportions(note, width)


def stitch_together(obverse_image, reverse_image, padding_height=6, margin_width=5):
    """
    stitch parts int final image

    :param obverse_image: path to obverse
    :param reverse_image: path to reverse
    :param padding_height: height of horizontal padding
    :param margin_width: width of vertical margin
    :return:              result image
    """
    strip_up, note_up = process_scan(obverse_image)
    strip_down, note_down = process_scan(reverse_image)

    padding_horizontal = np.zeros((padding_height, strip_up.shape[1], 3))

    result = np.concatenate((
        padding_horizontal,
        strip_up,
        padding_horizontal,
        note_up,
        padding_horizontal,
        note_down,
        padding_horizontal,
        strip_down,
        padding_horizontal,
    ))

    """
    add vertical margin
    """
    margin_vertical = np.zeros((result.shape[0], margin_width, 3))
    result = np.concatenate((margin_vertical, result, margin_vertical), axis=1)

    return result


if __name__ == "__main__":
    files = glob.glob('images\\*[!r].bmp')
    print(files)
    for name in files:
        print(name)
        reverse_name = name.split('.')
        reverse_name = reverse_name[0] + 'r.' + reverse_name[1]
        result = stitch_together(name, reverse_name)
        name = name.split('\\')[1].split('.')
        misc.imsave('results\\' + name[0] + '.jpg' , result)
