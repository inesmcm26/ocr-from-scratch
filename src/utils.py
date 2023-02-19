# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def read(image_path, label_path):
    # get the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get the image shape
    image_h, image_w, _ = image.shape
    # resize
    # TODO: resize maintaining aspect ratio
    # add padding in the direction that is missing pixels
    # adjust x, y, w and h accordingly
    image = cv2.resize(image, (448, 448))

    # draw_grids(image)

    image = image / 255

    # S x S x (B * 5 + C) = 15 x 15 x (2 * 5 + 1) = 15 x 15 x 11
    label_matrix = np.zeros([15, 15, 11])

    for line in open(label_path, 'r'):
        # read the line
        line = line.strip('\ufeff').strip()
        # split the line
        l = line.split(',')
        l = np.array(l[:8], dtype=np.int)
        xmin = l[0]
        ymin = l[1]
        xmax = l[4]
        ymax = l[5]

        # get the class
        cls = 0

        # get the bounding box coordinated in yolo format
        x = (xmin + xmax) / 2 / image_w
        y = (ymin + ymax) / 2 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h

        # get the cell coordinates
        loc = [15 * x, 15 * y]

        loc_i = int(loc[1])
        loc_j = int(loc[0])

        # get the offset
        y = loc[1] - loc_i
        x = loc[0] - loc_j


        # label_matrix[loc_i, loc_j, :] = [cls == 1, x, y, h, w, confidence]
        if label_matrix[loc_i, loc_j, 5] == 0:
            label_matrix[loc_i, loc_j, cls] = 1
            label_matrix[loc_i, loc_j, 1:5] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 5] = 1  # confidence

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