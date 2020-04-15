#!/usr/bin/python
import sys
from skimage import io, measure
from matplotlib import pylab as plt
from skimage.feature import corner_harris, canny, corner_peaks
from skimage.transform import probabilistic_hough_line, rotate, rescale
from skimage.morphology import erosion, dilation, closing
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.measure import find_contours, approximate_polygon
import math
import numpy as np

io.use_plugin('matplotlib')

accuracy = 0.99
number_of_points = 80


# znajdywacz podstawy drugą metodą
def find_base2(img):
    data = canny(img, sigma=3)
    data = dilation(data)
    lines = probabilistic_hough_line(data, threshold=50, line_length=1,
                                     line_gap=5)
    max_distance = 0
    base_line = ()
    for line in lines:
        distance = math.sqrt((line[1][0] - line[0][0]) ** 2 + (line[1][1] - line[0][1]) ** 2)
        if distance > max_distance:
            max_distance = distance
            base_line = line
    return base_line


# znajdywacz podstawy pierwsza metodą
def find_base(img):
    contours = find_contours(img, 0)
    max_distance = 0
    base_line = ()
    coords = approximate_polygon(contours[0], tolerance=3)
    for i in range(len(coords) - 1):
        distance = math.sqrt((coords[i + 1][0] - coords[i][0]) ** 2 + (coords[i + 1][1] - coords[i][1]) ** 2)
        if distance > max_distance:
            if find_if_base_ok(img, [[coords[i + 1][1], coords[i + 1][0]], [coords[i][1], coords[i][0]]]) > 0:
                max_distance = distance
                base_line = [[coords[i + 1][1], coords[i + 1][0]], [coords[i][1], coords[i][0]]]
    return base_line


def find_furthest_bottom(img):
    for i in range(len(img) - 1, 0, -1):
        if img[i][len(img[0]) // 2] > 0:
            return i
    return -1


def midpoint(line):
    return [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]


def find_furthest_left(img):
    for i in range(len(img[0])):
        for j in range(len(img)):
            if img[j][i] > 0:
                return i
    return -1


def find_furthest_right(img):
    for i in range(len(img[0]) - 1, 0, -1):
        for j in range(len(img)):
            if img[j][i] > 0:
                return i
    return -1


def find_furthest_top(img):
    for j in range(len(img)):
        for i in range(len(img[0])):
            if img[j][i] > 0:
                return j
    return -1


def resizer(img):
    top = find_furthest_top(img)
    bottom = find_furthest_bottom(img)

    vertical = bottom - top

    right = find_furthest_right(img)
    left = find_furthest_left(img)

    horizontal = right - left

    vertical_scale = len(img) / vertical
    horizontal_scale = len(img[0]) / horizontal

    img = img[np.max([0, top - 1]):np.min([len(img) - 1, bottom + 1]),
          np.max([0, left - 1]):np.min([len(img[0]) - 1, right + 1])]

    img = rescale(img, np.max([vertical_scale, horizontal_scale]), anti_aliasing=False)
    return img


# sprawdzam po której stronie podstawy jest figura
def orientation_check(img, point):
    check = math.floor(point[1]) + 5
    if check > len(img) or check < 0:
        return 0
    if img[check][math.floor(point[0])] > 0:
        return 1
    else:
        return 0


# obraca do pionu
def rotator(img, points, perfect=False):
    if (points[1][0] - points[0][0]) == 0 or (points[1][1] - points[0][1]) == 0:
        slope = 0
    else:
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    if slope == 0 and (points[1][1] - points[0][1]) != 0:
        deg = 90
    else:
        deg = math.degrees(math.atan(slope))
    img = rotate(img, deg, resize=True)

    line = find_base2(img)
    point = midpoint(line)
    if perfect == False:
        orientation = orientation_check(img, point)
        if orientation > 0:
            img = rotate(img, 180)
    return img


def width_detection(img, middle):
    sides = [0, 0]
    sides[0] = find_furthest_left(img)
    sides[1] = find_furthest_right(img)
    return sides


def find_mid_points(sides):
    columns = list(map(int, np.linspace(sides[0], sides[1], number_of_points + 2)))
    columns = columns[2:-1]
    return columns


# znajduje punkty na cięciu
def measure_edges(img, columns):
    points = []
    for col in columns:
        for i in range(img.shape[0]):
            if img[i][col] > accuracy:
                points.append([col, i])
                break
    return points


def compute_distances(cut_points1, cut_points2):
    distances = []
    j = len(cut_points2) - 1
    for i in range(len(cut_points1)):
        distance = cut_points1[i][1] + cut_points2[j][1]
        j -= 1
        distances.append(distance)
    return distances


def preference_hacker(preferences, doubles):
    for i in range(len(preferences)):
        if doubles[i] == 0:
            temp = np.where(preferences[preferences[i][0]] == i)
            preferences[preferences[i][0]][0], preferences[preferences[i][0]][temp] = preferences[preferences[i][0]][
                                                                                          temp], \
                                                                                      preferences[preferences[i][0]][0]
    return preferences


def distance_comparator(points_all):
    preferences = []
    doubles = np.zeros(len(points_all))
    i = 0
    for points1 in points_all:
        scores = []
        for points2 in points_all:
            if points1 != points2:
                distances = compute_distances(points1, points2)
                scores.append(np.std(distances))
        preference = np.argsort(scores)
        for j in range(len(preference)):
            if preference[j] >= i:
                preference[j] += 1
        preferences.append(preference)
        doubles[preference[0]] += 1
        i += 1
    return preferences, doubles


# znajduje prostopadłe do prostej przechodzącej przez punkty z inputu, proste są zapisane jako a oraz b
def find_perpendicular(points):
    if ((points[1][0] - points[0][0]) == 0 or (points[1][1] - points[0][1]) == 0):
        slope = 0
    else:
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        slope = -1 / slope
    # slope to a w funkcji liniowej

    # to jest przygotowanko do generowania punktów interpolowanych pomiędzy tymi dwoma punktami z inputu
    pointNum = 20
    diff_X = points[1][0] - points[0][0]
    diff_Y = points[1][1] - points[0][1]
    interval_X = diff_X / (pointNum + 1)
    interval_Y = diff_Y / (pointNum + 1)
    point_list = []
    lines = []
    # tu generuje punkty pomiędzy tymi punktami z inputu (czyli punkty na podstawie) i proste prostopadłe, które
    # przez nie przechodzą i po których znajdziemy te punkty z przecięcia
    for i in range(1, pointNum + 1):
        point_list.append([points[0][0] + interval_X * i, points[0][1] + interval_Y * i])
        lines.append([slope, point_list[i - 1][1] - (slope * point_list[i - 1][0])])
    return lines, point_list


# sprawdzam po której stronie lini jest figura
def orientation_check_old(img, line, point):
    # to plus 3 zapewnia, że na pewno już się zaczęła ta figura, bo te punkty mogą być czasami lekko oddalone od figury
    # ale można w sumie zwiększyć jak nie będzie łapać
    y = math.floor(line[0] * (math.floor(point[0]) + 3) + line[1])
    if y > len(img) or y < 0 or math.floor(point[0]) + 3 > len(img[0]):
        return 0
    if img[y][math.floor(point[0]) + 1] > 0:
        return 1
    else:
        return 0


# to jest w sumie skomplikowaność, lines to te prostopadłe do podstawy linie, a points to punkty na podstawie,
# przez które te proste przechodzą

# w skrócie od punktu podstawy ide po odpowiadającej mu funkcji liniowej aż nie skończy się obrazek
def find_furthest_point(img, lines, points):
    # orientacje sprawdzam na prostopadłej przechodzącej przez mniej więcej środek podstawy
    orientation = orientation_check_old(img, lines[math.floor(len(lines) / 2)], points[math.floor(len(lines) / 2)])
    for n in range(len(lines)):
        BAD = 0
        line = lines[n]
        if orientation != 1:
            for_range = range(math.floor(points[n][0]) - 1, 0 - 1, -1)
        else:
            for_range = range(math.ceil(points[n][0]) + 1, len(img[0]), 1)
        # dla każdej prostej jadę po niej od punktu aż do końca figury
        for i in for_range:
            if orientation != 1:
                y = math.floor(line[0] * i + line[1])
            else:
                y = math.ceil(line[0] * i + line[1])
            # jak się kończy ten biały obrazek, albo fizycznie obrazek to uznaje, że dotarłem do końca
            if y > 0 and y < len(img) and img[y][i] == 1 and BAD != 1:
                BAD += 1
            elif y > 0 and y < len(img) and BAD == 1 and img[y][i] == 0:
                BAD += 1
            if BAD > 2:
                return 0
    return 1


def find_if_base_ok(img, line):
    lines, point_list = find_perpendicular(line)
    result = find_furthest_point(img, lines, point_list)
    return result


def print_result(result):
    n = 0
    for r in result:
        for i in r:
            print(i, end=" ")
        print()
        n += 1


def processing(data):
    line = find_base(data)
    data = rotator(data, line)
    data = resizer(data)
    middle = [len(data[0]) // 2, find_furthest_bottom(data)]
    sides = width_detection(data, middle)
    columns = find_mid_points(sides)
    cut_points = measure_edges(data, columns)
    return cut_points, data


if __name__ == "__main__":
    _, path_param, instance_count = sys.argv
    instance_count = int(instance_count)
    points_all = []
    checkpoint = []
    contours = []
    for img_nr in range(instance_count):
        nazwa_pliku = "{}/{}.png".format(path_param, img_nr)
        data = io.imread(nazwa_pliku)
        cut_points, data = processing(data)
        points_all.append(cut_points)
    result, doubles = distance_comparator(points_all)
    result = preference_hacker(result, doubles)
    print_result(result)
