# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

GRID_H = 16 # yolo output grid height
GRID_W = 16 # yolo output grid width
B = 1 # nr of boxes
C = 1 # nr of classes
IMG_SHAPE = 512 # input height and width

def read(image_path, label_path):
    # get the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get the image shape
    image_h, image_w, _ = image.shape

    width_sl = IMG_SHAPE / image_w
    height_sl = IMG_SHAPE / image_h

    # resize
    # TODO: resize maintaining aspect ratio
    # add padding in the direction that is missing pixels
    # adjust x, y, w and h accordingly
    image = cv2.resize(image, (IMG_SHAPE, IMG_SHAPE))

    # draw_grids(image)

    image = image / 255

    # S x S x (B * 5 + C) = 16 x 16 x (1 * 5 + 0) = 16 x 16 x 5
    label_matrix = np.zeros((GRID_H, GRID_W, C, (B * 5)))

    for line in open(label_path, 'r'):
        # read the line
        line = line.strip('\ufeff').strip()
        # split the line
        l = line.split(',')
        l = np.array(l[:8], dtype=np.int)

        xmin = l[0] * width_sl
        ymin = l[1] * height_sl
        xmax = l[4] * width_sl
        ymax = l[5] * height_sl

        # get the bounding box coordinated in yolo format
        x = (xmin + xmax) / 2 / IMG_SHAPE
        y = (ymin + ymax) / 2 / IMG_SHAPE

        w = (xmax - xmin) / IMG_SHAPE
        h = (ymax - ymin) / IMG_SHAPE

        # get the cell coordinates
        loc = [GRID_W * x, GRID_H * y]

        loc_i = int(loc[1])
        loc_j = int(loc[0])

        # get the offset
        y = loc[1] - loc_i
        x = loc[0] - loc_j


        # label_matrix[loc_i, loc_j, :] = [pc, x, y, h, w]
        label_matrix[loc_i, loc_j, 0, 0] == 1 # pc
        label_matrix[loc_i, loc_j, 0, 1:5] = [x, y, w, h] # bounding box

    return image, label_matrix
        

def draw_grids(image):
    gridLineWidth = 29.866666666666667
    fig = plt.figure(figsize=(float(image.shape[0])/gridLineWidth,float(image.shape[1])/gridLineWidth), dpi=gridLineWidth)
    
    axes=fig.add_subplot(111)

    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    axes.set_aspect('equal')

    gridInterval=29.866666666666667
    location = plticker.MultipleLocator(base=gridInterval)
    axes.xaxis.set_major_locator(location)
    axes.yaxis.set_major_locator(location)

    axes.grid(which='major', axis='both', linestyle='-', color='k')
    axes.imshow(image)

    return axes

def resize(size, image, anns):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    new_anns = []
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.float64)
        poly *= scale
        new_ann = {'poly': poly.tolist(), 'text': ann['text']}
        new_anns.append(new_ann)
    return padimg, new_anns