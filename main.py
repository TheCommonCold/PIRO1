from skimage import io, measure
from matplotlib import pylab as plt
from skimage.feature import corner_harris, canny, corner_peaks
from skimage.transform import probabilistic_hough_line, rotate
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
        temp = np.array(contour[i])
        if len(temp) > 2:
            ax.plot(temp[:,0],temp[:,1], 'ro', markersize=5, linewidth=2)
        else:
            p0, p1 = temp
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]), markersize=10, linewidth=5)
        ax.set_title(nazwa)
        ploty.append(ax)
        io.imshow(checkpoint[i])

    return ploty

#znajdywacz podstawy drugą metodą
def find_base2(img):
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

#znajdywacz podstawy pierwsza metodą
def find_base(img):
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

#interpoluje punkty pomiędzy końcami podstawy
def find_middle_points(points):
    pointNum = 20
    diff_X = points[1][0]- points[0][0]
    diff_Y = points[1][1] - points[0][1]
    interval_X = diff_X / (pointNum + 1)
    interval_Y = diff_Y / (pointNum + 1)

    point_list = []

    for i in range(1,pointNum+1):
        point_list.append([points[0][0] + interval_X * i, points[0][1] + interval_Y*i])
    return point_list

#sprawdzam po której stronie podstawy jest figura
def orientation_check(img, point):
    check = math.floor(point[1])+5
    if check>len(img) or check<0:
        return 0
    if img[check][math.floor(point[0])] > 0:
        return 1
    else:
        return 0

#znajduje punkty na cięciu
def measure_edges(img, points, orientation):
    result = []
    for point in points:
        if orientation != 1:
            for_range = range(math.floor(point[1])-2, 0, -1)
            x = math.floor(point[0])
        else:
            for_range = range(math.ceil(point[1])+2,len(img))
            x = math.ceil(point[0])
        for i in for_range:
            if img[i,x]>0:
                continue
            else:
                result.append([x,i])
                break
    return result


#obraca do pionu
def rotator(img, points):
    if ((points[1][0] - points[0][0]) == 0 or (points[1][1] - points[0][1]) == 0):
        slope = 0
    else:
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    deg = math.degrees(math.atan(slope))
    return rotate(img, deg)

if __name__ == "__main__":
    how_many_in_folder = [6, 20, 20, 20, 20, 200, 20, 100]
    for set_nr in range(9):
        checkpoint = []
        contours = []
        for img_nr in range(how_many_in_folder[set_nr]):

            nazwa_pliku = "set{}/{}.png".format(set_nr, img_nr)
            print(nazwa_pliku)
            data = io.imread(nazwa_pliku)

            line = find_base(data)
            data = rotator(data,line)
            #znajduje nową podstawe, ten algos lepiej działa niż ten pierwszy w tym przypadku
            line = find_base2(data)
            points = find_middle_points(line)
            orientation = orientation_check(data,points[math.floor(len(points)/2)])
            cut_points = measure_edges(data,points,orientation)

            checkpoint.append(data)
            contours.append(cut_points)

        display(checkpoint, '', contours)
        io.show()

