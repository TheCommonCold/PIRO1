from skimage import io, measure
from matplotlib import pylab as plt
from skimage.feature import corner_harris, canny, corner_peaks
from skimage.transform import probabilistic_hough_line
from skimage.morphology import erosion, dilation, closing
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.measure import find_contours, approximate_polygon
import math
import numpy as np
io.use_plugin('matplotlib')

def display(checkpoint, nazwa, contour):
    rows = len(checkpoint)
    columns = 1
    fig = plt.figure(figsize=(5, rows * 5))
    ploty = []
    for i in range(rows):
        ax = fig.add_subplot(rows, columns, i + 1)
        if len(contour[i]) > 2:
            ax.plot(contour[i][:, 1], contour[i][:, 0], 'ro', markersize=5, linewidth=2)
        else:
            p0, p1 = contour[i]
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]), markersize=10, linewidth=5)
        ax.set_title(nazwa)
        ploty.append(ax)
        io.imshow(checkpoint[i])

    return ploty


def findBase2(img):
    data = canny(img, sigma=3)
    data = dilation(data)
    lines = probabilistic_hough_line(data, threshold=50, line_length=1,
                                     line_gap=5)
    max_distance = 0
    base_line = ()
    for line in lines:
        distance = math.sqrt((line[1][0] - line[0][0])**2 + (line[1][1] - line[0][1])**2)
        if distance > max_distance:
            max_distance = distance
            base_line = line
    return base_line

def findBase(img):
    contours = find_contours(img, 0)
    max_distance = 0
    base_line = ()
    coords = approximate_polygon(contours[0], tolerance=2.5)
    for i in range(len(coords)-1):
        distance = math.sqrt((coords[i+1][0] - coords[i][0])**2 + (coords[i+1][1] - coords[i][1])**2)
        if distance > max_distance:
            max_distance = distance
            base_line = [[coords[i+1][1],coords[i+1][0]],[coords[i][1],coords[i][0]]]
    return base_line

if __name__ == "__main__":
    how_many_in_folder = [6, 20, 20, 20, 20, 200, 20, 100]
    for set_nr in range(9):
        checkpoint = []
        contours = []
        for img_nr in range(how_many_in_folder[set_nr]):

            nazwa_pliku = "set{}/{}.png".format(set_nr, img_nr)
            print(nazwa_pliku)
            data = io.imread(nazwa_pliku)

            line = findBase(data)
            checkpoint.append(data)
            contours.append(line)

            line = findBase2(data)
            checkpoint.append(data)
            contours.append(line)


        display(checkpoint, '', contours)
        io.show()

