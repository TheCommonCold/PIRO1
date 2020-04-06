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

#znajdywacz podstawy drugą metodą, nie istotne
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

#znajdywacz podstawy pierwsza metodą, nie istotne
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

#znajduje prostopadłe do prostej przechodzącej przez punkty z inputu, proste są zapisane jako a oraz b
def find_perpendicular(points):
    if ((points[1][0] - points[0][0]) == 0 or (points[1][1] - points[0][1]) == 0):
        slope = 0
    else:
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        slope = -1/slope
    #slope to a w funkcji liniowej

    #to jest przygotowanko do generowania punktów interpolowanych pomiędzy tymi dwoma punktami z inputu
    pointNum = 20
    diff_X = points[1][0]- points[0][0]
    diff_Y = points[1][1] - points[0][1]
    interval_X = diff_X / (pointNum + 1)
    interval_Y = diff_Y / (pointNum + 1)

    point_list = []
    lines = []
    #tu generuje punkty pomiędzy tymi punktami z inputu (czyli punkty na podstawie) i proste prostopadłe, które
    #przez nie przechodzą i po których znajdziemy te punkty z przecięcia
    for i in range(1,pointNum+1):
        point_list.append([points[0][0] + interval_X * i, points[0][1] + interval_Y*i])
        lines.append([slope, point_list[i - 1][1] - (slope * point_list[i - 1][0])])
    return lines, point_list

#sprawdzam po której stronie lini jest figura
def orientation_check(img, line, point):
    # to plus 3 zapewnia, że na pewno już się zaczęła ta figura, bo te punkty mogą być czasami lekko oddalone od figury
    # ale można w sumie zwiększyć jak nie będzie łapać
    y = math.floor(line[0] * (math.floor(point[0])+3) + line[1])
    if y>len(img) or y<0 or math.floor(point[0])+3>len(img[0]):
        return 0
    if img[y][math.floor(point[0])+1] > 0:
        return 1
    else:
        return 0


# to jest w sumie skomplikowaność, lines to te prostopadłe do podstawy linie, a points to punkty na podstawie,
# przez które te proste przechodzą

# w skrócie od punktu podstawy ide po odpowiadającej mu funkcji liniowej aż nie skończy się obrazek
def find_furthest_point(img, lines,points):
    result = []
    #orientacje sprawdzam na prostopadłej przechodzącej przez mniej więcej środek podstawy
    orientation = orientation_check(img, lines[math.floor(len(lines)/2)],points[math.floor(len(lines)/2)])
    for n in range(len(lines)):
        line = lines[n]
        if orientation!=1:
            for_range = range(math.floor(points[n][0])-1,0, -1)
        else:
            for_range = range(math.ceil(points[n][0])+1, len(img[0]), 1)
        # dla każdej prostej jadę po niej od punktu aż do końca figury
        for i in for_range:
            if orientation!=1:
                y = math.floor(line[0]*i+line[1])
            else:
                y = math.ceil(line[0] * i + line[1])
            #jak się kończy ten biały obrazek, albo fizycznie obrazek to uznaje, że dotarłem do końca
            if y>len(img) or y<0:
                result.append([i, y])
                break
            if img[y][i]>0:
                continue
            else:
                result.append([i, y])
                break
    return result

def find_cut_points(img, line):
    lines, point_list = find_perpendicular(line)
    points = find_furthest_point(img, lines, point_list)
    return points

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

            checkpoint.append(data)
            points = find_cut_points(data,line)
            contours.append(points)

        display(checkpoint, '', contours)
        io.show()

