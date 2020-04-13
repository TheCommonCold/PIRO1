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

accuracy = 0.2
number_of_points = 80

def display(img, line , points, name=""):
    fig = plt.figure(figsize=(5, 5))
    ploty = []
    ax = fig.add_subplot(1, 1, 1)
    if line:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), markersize=10, linewidth=5)
    if points:
        temp = np.array(points)
        ax.plot(temp[:, 0], temp[:, 1], 'ro', markersize=5, linewidth=2)
    ax.set_title(name)
    ploty.append(ax)
    io.imshow(img)
    return ploty

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
            max_distance = distance
            base_line = [[coords[i + 1][1], coords[i + 1][0]], [coords[i][1], coords[i][0]]]
    return base_line

def find_furthest_bottom(img):
    for i in range(len(img)-1,0,-1):
        if img[i][len(img[0])//2]>0:
            return i
    return -1

def midpoint(line):
    return [(line[0][0]+line[1][0])/2, (line[0][1]+line[1][1])/2]

def find_furthest_left(img):
    for i in range(len(img[0])):
        for j in range(len(img)):
            if img[j][i]>0:
                return i
    return -1

def find_furthest_right(img):
    for i in range(len(img[0])-1,0,-1):
        for j in range(len(img)):
            if img[j][i]>0:
                return i
    return -1

def find_furthest_top(img):
    for j in range(len(img)):
        for i in range(len(img[0])):
            if img[j][i]>0:
                return j
    return -1

def resizer(img):
    top = find_furthest_top(img)
    bottom = find_furthest_bottom(img)

    vertical = bottom-top

    right = find_furthest_right(img)
    left = find_furthest_left(img)

    horizontal = right-left

    vertical_scale = len(img)/vertical
    horizontal_scale = len(img[0])/horizontal

    img = img[np.max([0,top-1]):np.min([len(img)-1,bottom+1]),np.max([0,left-1]):np.min([len(img[0])-1,right+1])]

    img = rescale(img, np.max([vertical_scale,horizontal_scale]), anti_aliasing=False)
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
def rotator(img, points,perfect = False):
    if (points[1][0] - points[0][0]) == 0 or (points[1][1] - points[0][1]) == 0:
        slope = 0
    else:
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    if slope == 0 and (points[1][1] - points[0][1])!=0:
        deg = 90
    else:
        deg = math.degrees(math.atan(slope))
    img = rotate(img, deg, resize=True)

    line = find_base2(img)
    point = midpoint(line)
    if perfect==False:
        orientation = orientation_check(img, point)
        if orientation > 0:
            img = rotate(img, 180)
    return img



def artur(data):
    line = find_base2(data)
    # 10 pixeli nad podstawą
    line_h = line[0][1] - 10
    line = [0, 0]
    for i in range(data.shape[0]):
        if data[line_h][i]:
            line[0] = i
            break
    for i in range(data.shape[1] - 1, -1, -1):
        if data[line_h][i]:
            line[1] = i
            break
    line = line if line[0] < line[1] else [line[1], line[0]]
    # TODO parametrise
    line[0] += data.shape[1] // 50
    line[1] -= data.shape[1] // 50
    # TODO parametrise
    number_of_points = 80
    columns = list(map(int, np.linspace(line[0], line[1], number_of_points)))
    points = []
    for col in columns:
        for i in range(data.shape[0]):
            if data[i][col]:
                points.append([col, i])
                break
    # print(pochodne)
    return points

def final_artur(points):
    result = []
    # pochodne = []
    # for p in points:
    #     pochodne_singular = []
    #     for i in range(len(p) - 1):
    #         [x1, y1] = p[i]
    #         [x2, y2] = p[i + 1]
    #         dy = y2 - y1
    #         dx = x2 - x1
    #         pochodne_singular.append(dy / dx)
    #     pochodne.append(pochodne_singular)
    sums_all = [[] for _ in range(len(points))]

    for i in range(len(points)):
        p1 = points[i]
        for j in range(i + 1, len(points)):
            if i == j: continue
            p2 = points[j]
            sums = []
            for k in range(len(p1)):
                sums.append(p1[k][1] + p2[len(p1) - k - 1][1])
            std = np.std(sums)
            sums_all[i].append([j, std])
            sums_all[j].append([i, std])
    for l in sums_all:
        newlist = sorted(l, key=lambda x: x[1])
        only_nums = list(map(lambda x: x[0], newlist))
        result.append(only_nums)
    return result
    # for i in range(len(pochodne)):
    #     p1 = pochodne[i]
    #     print(p1)
    #     errors = []
    #     for j in range(len(pochodne)):
    #         if i == j: continue
    #         p2 = pochodne[j]
    #         error = 0.
    #         for k in range(len(p1)):
    #             error += (p1[k] + p2[len(p1) - k - 1]) ** 2
    #         errors.append([error, j])
    #     print(errors)
    #     # print(errors)


def width_detection(img, middle):
    sides = [0, 0]
    sides[0] = find_furthest_left(img)
    sides[1] = find_furthest_right(img)
    return sides

def find_mid_points(sides):
    columns = list(map(int, np.linspace(sides[0], sides[1], number_of_points+2)))
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
    j=len(cut_points2)-1
    for i in range(len(cut_points1)):
        distance = cut_points1[i][1]+cut_points2[j][1]
        j-=1
        distances.append(distance)
    return distances

def preference_hacker(preferences, doubles):
    print(doubles)
    print_result(preferences)
    for i in range(len(preferences)):
        if doubles[i]==0:
            temp=np.where(preferences[preferences[i][0]] == i)
            preferences[preferences[i][0]][0],preferences[preferences[i][0]][temp] = preferences[preferences[i][0]][temp], preferences[preferences[i][0]][0]
    return preferences

def distance_comparator(points_all):
    preferences = []
    doubles = np.zeros(len(points_all))
    i=0
    for points1 in points_all:
        scores = []
        for points2 in points_all:
            if points1 != points2:
                distances = compute_distances(points1,points2)
                scores.append(np.std(distances))
        preference = np.argsort(scores)
        for j in range(len(preference)):
            if preference[j]>=i:
                preference[j] += 1
        preferences.append(preference)
        doubles[preference[0]]+=1
        i += 1
    return preferences,doubles

def retardo_comperator(points_all):
    preferences = []
    i = 0
    retard = []
    for points in points_all:
        score = []
        for i in range(len(points)-1):
            if points[i+1][1]>points[i][1]:
                score.append(1)
            else:
                score.append(0)
            retard.append(score)
    for score1 in retard:
        print(score1)
        scores = []
        for score2 in retard:
            if score1!=score2:
                temp = 0
                for i in score1:
                    for j in reversed(score2):
                        if i == j:
                            temp+=1
                scores.append(temp)
        preference = np.flip(np.argsort(scores))
        for j in range(len(preference)):
            if preference[j]>=i:
                preference[j] += 1
        preferences.append(preference)
        i += 1
    return preferences

def print_result(result):
    n=0
    for r in result:
        print("Obrazek",n, ":", end=" ")
        for i in r:
            print(i, end=" ")
        print()
        n+=1


def processing(data, debug_name=""):
    line = find_base(data)
    data = rotator(data, line)
    data = resizer(data)
    middle = [len(data[0])//2,find_furthest_bottom(data)]
    sides = width_detection(data, middle)
    columns = find_mid_points(sides)
    cut_points = measure_edges(data, columns)
    #display(data,False,cut_points,debug_name)
    return cut_points, data

if __name__ == "__main__":
    how_many_in_folder = [6, 20, 20, 20, 20, 200, 200, 20,100]
    wypis_na_koniec=""
    for set_nr in range(0,9):
        f = open("set{}/correct.txt".format(set_nr), "r")
        correct = list(map(int, f.read().split('\n')[:-1]))
        points_all = []
        checkpoint = []
        contours = []
        for img_nr in range(how_many_in_folder[set_nr]):
            nazwa_pliku = "set{}/{}.png".format(set_nr, img_nr)
            print(nazwa_pliku)
            data = io.imread(nazwa_pliku)

            cut_points, data = processing(data,"set{}/{}.png".format(set_nr, img_nr))
            if len(cut_points) == 0:
                print("DEBUG",cut_points)
                display(data,False,False)
                io.show()
            #cut_points = artur(data)
            points_all.append(cut_points)

        io.show()
        result, doubles = distance_comparator(points_all)
        result = preference_hacker(result, doubles)
        print_result(result)
        sum_of_points = 0.
        for i in range(len(correct)):
            for j in range(len(result[i])):
                if result[i][j] == correct[i]:
                    sum_of_points += (1 / (1 + j))
                    break
        wypis=str(sum_of_points)+ ' na '+ str(len(correct))
        print(wypis)
        wypis_na_koniec+=wypis+'\n'
    print(wypis_na_koniec)
