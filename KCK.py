from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure, measure, feature
from skimage.transform import rescale
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from skimage.filters import *
from matplotlib import pylab as plt
from skimage.morphology import watershed
import scipy as sci
import numpy as np
from scipy import ndimage as ndi
from numpy import array
from skimage.measure import label
from skimage import data, util
from matplotlib import colors
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage import transform as tf
import operator
from skimage.transform import rotate

io.use_plugin('matplotlib')


# Wymiary:
# Plansza = 263
# Kolko  = 24
# maly szesciokat = 45

def wyswietl(checkpoint, nazwa):
    rows = len(checkpoint)
    columns = 1
    fig = plt.figure(figsize=(5, rows * 5))
    ploty = []
    for i in range(rows):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_title(nazwa)
        ploty.append(ax)
        io.imshow(checkpoint[i])

    return ploty


# zostawia kolor(w hsv) pomiędzy min i max i zmienia reszte na 0
def filter_colour(data, min, max, hsv):
    for array in data:
        for x in array:
            if x[hsv] < min or x[hsv] > max:
                x[0] = 0
                x[1] = 0
                x[2] = 0
    return data

def filter_colour_reverse(data, min1, max1):
    for array in data:
        for x in array:
            if x[0] > min1 and x[0] < max1 :
                x[0] = 0
                x[1] = 0
                x[2] = 0
    return data


# zmienia kolor(w hsv) pomiędzy min i max na 1, a reszte na zero
def filter_colour_hard(data, min, max, hsv):
    output = []
    for array in data:
        temparray = []
        for x in array:
            if x[hsv] < min or x[hsv] > max:
                temparray.append(0)
            else:
                temparray.append(1)
        output.append(temparray)
    return output


# tworzy maske jedynek na wodzie
# tworzy maske jedynek na wodzie
def background_removal(data):
    checkpoint = []
    p1, p2 = np.percentile(data, (1, 92))
    data = exposure.rescale_intensity(data, in_range=(p1, p2))
    data = rgb2hsv(data)
    checkpoint.append(hsv2rgb(data))
    data = np.array(filter_colour(data, 0.5, 0.7, 0))
    data = np.array(filter_colour(data, 0.1, 1, 1))
    data = np.array(rgb2hsv(filter_colour(hsv2rgb(data), 0.2, 1, 2)))
    data = np.array(filter_colour_hard(data, 0.1, 1, 2))
    data = mp.dilation(data)
    checkpoint.append(data)
    contours = measure.find_contours(data, 0.2)

    return checkpoint, contours


# matematyczny argmin argmax
def arg_min_max(lista):
    # zwarca argumenty min i max danej listy
    max = min = lista[0]
    argmin = 0
    argmax = 0
    for k in range(len(lista)):
        if lista[k] < min:
            min = lista[k]
            argmin = k
        elif (lista[k] > max):
            max = lista[k]
            argmax = k
    return argmin, argmax


# tworzy maske zer w miejscu lądu
def outer_removal(img):
    img2 = img.copy()
    maxJ = len(img[0])
    for i in range(len(img)):
        for j in range(maxJ):
            if img[i][j] == 1:
                break
            img2[i][j] = 1
        for j in range(1, maxJ):
            if img[i][maxJ - j] == 1:
                break
            img2[i][maxJ - j] = 1
    return img2


# zostawia tylko wyspe (usuwa wode i stół)
def leave_only_island(img, mask):
    # mask = dilation_loop(mask, 2)
    img2 = img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j]:
                img2[i][j] = np.array([0, 0, 0])
    return img2


# zostawia tylko jeden kolor (niepoprawny grayscale)
def leave_only_one_color(img, color):
    if color == "red" or color == "r":
        color = 0
    elif color == "green" or color == "g":
        color = 1
    elif color == "blue" or color == "b":
        color = 2
    return np.array([[(x[color]) for x in array] for array in img])


# zwraca minimalną i maksymalną wartość z macierzy
def minimaxi(img):
    mini = 2
    maxi = -1
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > maxi:
                maxi = img[i][j]
        if img[i][j] < mini:
            mini = img[i][j]
    return mini, maxi


# robi zakres od 0 to 1 z węższego
def ujednolic(img):
    mini, maxi = minimaxi(img)
    img = (img - mini) / (maxi - mini)
    return img


# przerabia na 0 i 1 zależnie od threshodu
def threshold(img, thr):
    # std = np.std(img)
    # var = np.var(img)
    # print(nazwa + " " + str(std) + " " + str(var))
    return img > thr


# Odwrotnie, bo czarny, a nie biały
def dilation(img):
    return mp.erosion(img)


# Odwrotnie, bo czarny, a nie biały
def erosion(img):
    return mp.dilation(img)


def erosion_loop(data, times):
    for i in range(times):
        data = mp.erosion(data)
    return data


def dilation_loop(data, times):
    for i in range(times):
        data = mp.dilation(data)
    return data


# daje numer największego contouru
def sort_contourow(contours):
    return sorted(contours, key=lambda x: polygon_area(x))


def top_contoury(contours, ile):
    output = []
    for i in range(len(contours) - ile, len(contours)):
        output.append(contours[i])
    return output


def centroid_z_konturu(contour, ax, colour):
    centroid = np.sum(contour, axis=0) / len(contour)
    q = np.random.uniform()
    c = colors.hsv_to_rgb([q, 1, 1])
    # ax.plot(contour[:, 1], contour[:, 0], linewidth=3, color=c)

    c = colors.hsv_to_rgb([colour, 1, 1])
    ax.plot(centroid[1], centroid[0], marker="o", color=c)


def kontury_do_srodkow(contours, ax, colour):
    for n, contour in enumerate(contours):
        if (len(contour) > 10):
            centroid_z_konturu(contour, ax, colour)


def usuwanko_punktow(contour):
    test_dist = 1
    delete_array = [];
    while (len(contour) > 6):
        for x in range(len(contour) - 1):
            for y in range(x, len(contour) - 1):
                dist = math.sqrt((contour[y][0] - contour[x][0]) ** 2 + (contour[y][1] - contour[x][1]) ** 2)
                if (dist < test_dist):
                    delete_array.append(y)
        while (len(contour) - len(delete_array) < 6):
            delete_array.pop()
        contour = np.delete(contour, delete_array, 0)
        print(len(contour))
        test_dist = test_dist + 1

    return contour


def srednia_kanalu(data, rgb):
    sum = 0
    n = 0
    for array in data:
        for x in array:
            if x[0] != 0 and x[1] != 0 and x[2] != 0:
                sum = sum + x[rgb]
                n = n + 1
    return sum / n

def pewnosc(data, max):
    sum = 0
    n=0
    for array in data:
        for x in array:
            sum = sum + x
            n = n + 1
    return (sum / n)/max


def rozciagnij_3_wartosci(image, first=True, second=True, third=True):
    img = image.copy()
    m = []
    l = []
    if first:
        l.append(0)
    if second:
        l.append(1)
    if third:
        l.append(2)

    for i in range(3):
        m.append([1, 0])
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in l:
                if img[i][j][k] < m[k][0]:
                    m[k][0] = img[i][j][k]
                if img[i][j][k] > m[k][1]:
                    m[k][1] = img[i][j][k]

    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in l:
                if (m[k][1] - m[k][0]) > 0:
                    img[i][j][k] = (img[i][j][k] - m[k][0]) / (m[k][1] - m[k][0])
    return img



def fill_contour_with_ones_stary(img):
    img = img.copy()
    for i in range(len(img)):
        k = 0
        l = 0
        for j in range(len(img[0])):
            if img[i][j] == 1:
                k = j
                break
        for j in range(len(img[0]) - 1, -1, -1):
            if img[i][j] == 1:
                l = j
                break
        for j in range(k, l, 1):
            img[i][j] = 1
    return img


def fill_contour_with_ones(img, srodek):
    srodek[0] = int(srodek[0])
    srodek[1] = int(srodek[1])

    return img


def doprecyzuj_narozniki(coord, img, dodawane, dokladnosc=20):
    to_check = [0, 0]
    for i in range(2):
        if dodawane[i] == 0:
            to_check[i] = 1
    changed = True
    while changed:
        print("coord:", coord, dodawane)
        changed = False
        for i in range(-dokladnosc, dokladnosc):
            # print("mem:",coord[0]+dodawane[0]+to_check[0]*i,coord[0]+dodawane[1]+to_check[1]*i)

            if img[coord[0] + dodawane[0] + to_check[0] * i][coord[1] + dodawane[1] + to_check[1] * i] == 1:
                coord = [coord[0] + dodawane[0] + to_check[0] * i, coord[1] + dodawane[1] + to_check[1] * i]
                changed = True
                break
    return coord


def doprecyzuj_narozniki_s(coord, img, dodawane):
    rangi = range(len(img))
    if dodawane[0] == 0:
        rangi = range(len(img[0]))
    changed = True
    while changed:
        changed = False
        lista = []
        for i in rangi:
            if dodawane[0] != 0:
                if img[coord[0] + dodawane[0]][i] == 1:
                    lista.append(i)
                    changed = True
            else:
                if img[i][coord[1] + dodawane[1]] == 1:
                    lista.append(i)
                    changed = True
        if changed:
            if dodawane[0] != 0:
                coord[0] += dodawane[0]
                coord[1] = int(np.mean(lista))
            else:
                coord[1] += dodawane[1]
                coord[0] = int(np.mean(lista))
    return coord


def contains_one(lista):
    for i, v in enumerate(lista):
        if v == 1:
            return i
    return -1


def coords_specify_pion(coords, img, dodac=1):
    lista = img[coords[0] + dodac]
    if dodac != 1:
        lista = reversed(lista)
    l = contains_one(lista)
    while l >= 0:
        coords[0] += dodac
        coords[1] = l
        l = contains_one(img[coords[0] + dodac])


def coords_specify_poziom(coords, img, dodac=1):
    lista = img[:, coords[1] + dodac]
    if dodac != 1:
        lista = reversed(lista)
    l = contains_one(lista)
    while l >= 0:
        coords[1] += dodac
        coords[0] = l
        l = contains_one(img[:, coords[1] + dodac])


def experimental_plotting(img, ax):
    srodek = [len(img) // 2, len(img[0]) // 2]
    coords = np.array([srodek, srodek, srodek, srodek])
    coords_specify_pion(coords[0], img, -1)
    coords_specify_poziom(coords[1], img, 1)
    coords_specify_pion(coords[2], img, 1)
    coords_specify_poziom(coords[3], img, -1)
    print(coords)
    ax.plot(coords[:, 1], coords[:, 0], color='green', marker='o', linestyle='dashed', linewidth=1, markersize=5)
    return img


def fill_coords(coords_all):
    coords = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    for i in range(4):
        coords[i] = coords_all[0]
    offset = 0
    for i in range(len(coords_all)):
        if coords_all[i][0] + offset < coords[0][0] or (
                coords_all[i][0] <= coords[0][0] and coords_all[i][1] < coords[0][1]):
            coords[0] = coords_all[i]
        if coords_all[i][1] + offset < coords[1][1] or (
                coords_all[i][1] <= coords[1][1] and coords_all[i][0] > coords[1][0]):
            coords[1] = coords_all[i]
        if coords_all[i][0] - offset > coords[2][0] or (
                coords_all[i][0] >= coords[2][0] and coords_all[i][1] > coords[2][1]):
            coords[2] = coords_all[i]
        if coords_all[i][1] - offset > coords[3][1] or (
                coords_all[i][1] >= coords[3][1] and coords_all[i][0] < coords[3][0]):
            coords[3] = coords_all[i]
    # if distance_between_two_points(coords[0], coords[3])<distance_between_two_points(coords[0], coords[1]):
    #     for i in range(len(coords_all)):
    #         if coords_all[i][0] <= coords[0][0] and coords_all[i][1] > coords[0][1]:
    #             coords[0] = coords_all[i]
    #             print("DEBUG")
    #         if coords_all[i][1] <= coords[1][1] and coords_all[i][0] < coords[1][0]:
    #             coords[1] = coords_all[i]
    #             print("DEBUG")
    #         if coords_all[i][0] >= coords[2][0] and coords_all[i][1] < coords[2][1]:
    #             coords[2] = coords_all[i]
    #             print("DEBUG")
    #         if coords_all[i][1] >= coords[3][1] and coords_all[i][0] < coords[3][0]:
    #             coords[3] = coords_all[i]
    #             print("DEBUG")
    return coords


def distance_between_two_points(coord1, coord2):
    return sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)


def cycle_coords_if_rectangle(coords, para):
    print("para", para)
    print("coords", coords)
    coords2 = coords.copy()
    if para == [0, 3]:
        para = [0, 1]
        for i in range(4):
            coords2[i] = coords[(i + 1) % 4].copy()
    return coords2, para


def cycle_coords_if_not_rect(coords, exception):
    coords2 = coords.copy()
    for i in range(4):
        coords2[i] = coords[(exception + i) % 4].copy()
    return coords2, 0


def define_para(dist, coords):
    para = [0, 1]
    for i in range(1, 3):
        if dist * 1.1 > distance_between_two_points(coords[0], coords[i]) > dist * 0.9:
            para[1] = i
            break
    return para


def define_przypadek(coords, dist):
    przypadek = 0
    for i in range(2):
        if distance_between_two_points(coords[0 + i], coords[2 + i]) < dist * 1.76:
            print("debug", distance_between_two_points(coords[0 + i], coords[2 + i]) / dist)
            przypadek = 1
    return przypadek


def define_exception(coords, dist):
    exception = 0
    for i in range(4):
        dobry = False
        for j in range(4):
            if i == j:
                continue
            if dist * 1.1 > distance_between_two_points(coords[j], coords[i]) > dist * 0.9:
                dobry = True
                break
        if not dobry:
            exception = i
            break
    return exception


def transform_if_rectangle(img, coords):
    left = 3000. - 150.
    right = 1000. + 150.
    top = 0.
    bot = 3000. - 56.
    dst = np.array([[left, top], [right, top], [right, bot], [left, bot]], dtype="float32")
    #
    tmp = coords[:, 0].copy()
    coords[:, 0] = coords[:, 1].copy()
    coords[:, 1] = tmp

    print("coords:", coords)
    print("dst:", dst)
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(dst, coords)
    warp = tf.warp(img, tform3, output_shape=(len(img), len(img[0])))
    return warp, dst


def transform_if_not_rect(img, coords):
    top = 0.
    midtop = 1450.
    bot = 3000. - 56.
    left = 300.
    midright = 1000. + 150.
    right = 3700.

    dst = np.array([[right, midtop], [midright, top], [left, midtop], [midright, bot]], dtype="float32")

    tmp = coords[:, 0].copy()
    coords[:, 0] = coords[:, 1].copy()
    coords[:, 1] = tmp

    print("coords:", coords)
    print("dst:", dst)
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(dst, coords)
    warp = tf.warp(img, tform3, output_shape=(len(img), len(img[0])))
    return warp, dst


def srodek_jedynek(img):
    srodek = [0, 0]
    liczba = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 1:
                liczba += 1
                srodek[0] += i
                srodek[1] += j
    srodek[0] /= liczba
    srodek[1] /= liczba
    return srodek


# suposedly calculates the area of any polygon
def polygon_area(corners):
    # przyklad: corners = [(2.0, 1.0), (4.0, 5.0), (7.0, 8.0)]
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def calc_min_dist_between_coords(coords):
    dist = distance_between_two_points(coords[0], coords[1])
    ind = 1
    for i in range(4):
        for j in range(i + 1, 4):
            dist2 = distance_between_two_points(coords[i], coords[j])
            if dist2 < dist:
                dist = dist2
                if i == 0:
                    ind = j
            print(dist2, i, j)
    return dist, [0, ind]


def kontury_debug(img):
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(img, 0.8)
    max = 0
    ind = 0
    for i, contour in enumerate(contours):
        p = polygon_area(contour)
        if p > max:
            max = p
            ind = i

    coords = fill_coords(contours[ind])
    dist, para = calc_min_dist_between_coords(coords)
    przypadek = define_przypadek(coords, dist)

    print("min dystans miedzy wierch:", dist)
    print("przypadek:", przypadek)

    if przypadek == 0:
        coords, para = cycle_coords_if_rectangle(coords, para)
        # for i in range(4):
        #     c = "+r"
        #     # if i in para:
        #     #     c = "+g"
        #     if i == 1:
        #         c = "+g"
        #     if i == 2:
        #         c = "+b"
        #     if i == 3:
        #         c = "+c"
        #     ax2.plot(coords[i, 1], coords[i, 0], c, markersize=15)
    exception=0
    if przypadek == 1:
        exception = define_exception(coords, dist)
        coords, exception = cycle_coords_if_not_rect(coords, exception)
        # for i in range(4):
        #     c = "+r"
        #     if i == exception:
        #         c = "+c"
        #     ax2.plot(coords[i, 1], coords[i, 0], c, markersize=15)

    return np.array(coords, dtype="float32"), przypadek,para,exception


def extract_tile(tile, img):
    left = 0.
    right = 550.
    top = 150.
    bot = 450.
    dst = np.array([[left, top], [right, top], [left, bot], [right, bot]], dtype="float32")

    coords = []
    for i in range(1, 5):
        coords.append(tile[i].copy())

    coords = np.array(coords, dtype="float32")
    tmp = coords[:, 0].copy()
    coords[:, 0] = coords[:, 1].copy()
    coords[:, 1] = tmp
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(dst, coords)
    warp = tf.warp(img, tform3, output_shape=(600, 550))
    return warp

def extract_town(town, img):
    left = 0.
    right = 300.
    top = 0.
    bot = 300.
    dst = np.array([[left, top], [right, top], [left, bot], [right, bot]], dtype="float32")

    offset=100.
    left = town[1]-offset
    right = town[1]+offset
    top = town[0]-offset
    bot = town[0]+offset
    coords = np.array([[left, top], [right, top], [left, bot], [right, bot]], dtype="float32")

    # tmp = coords[:, 0].copy()
    # coords[:, 0] = coords[:, 1].copy()
    # coords[:, 1] = tmp
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(dst, coords)
    warp = tf.warp(img, tform3, output_shape=(300, 300))
    return warp

def krojonko(img, checkpoint):
    checkpoint.append(img.copy())
    xs = np.array([725., 980., 1230., 1490., 1740., 2004., 2255., 2515., 2770., 3020., 3275.], dtype="float32")
    ys = np.array([290., 440., 730., 875., 1170., 1330., 1615., 1760., 2060., 2210., 2505., 2655.], dtype="float32")
    tile_coords = []
    town_coords = []
    road_coords = []
    for j in range(3):
        for i in range(3 + j):
            tile_coords.append([[ys[0 + 2 * j], xs[3 + i * 2 - j]], [ys[1 + 2 * j], xs[2 + i * 2 - j]],
                                [ys[1 + 2 * j], xs[4 + i * 2 - j]], [ys[2 + 2 * j], xs[2 + i * 2 - j]],
                                [ys[2 + 2 * j], xs[4 + i * 2 - j]], [ys[3 + 2 * j], xs[3 + i * 2 - j]]])
    for j in range(2):
        for i in range(4 - j):
            tile_coords.append([[ys[6 + 2 * j], xs[2 + 2 * i + j]],
                                [ys[7 + 2 * j], xs[1 + i * 2 + j]],
                                [ys[7 + 2 * j], xs[3 + i * 2 + j]],
                                [ys[8 + 2 * j], xs[1 + i * 2 + j]],
                                [ys[8 + 2 * j], xs[3 + i * 2 + j]],
                                [ys[9 + 2 * j], xs[2 + i * 2 + j]]])
    lista_pomocnicza_1 = [3,4,4,5,5,6]
    lista_pomocnicza_2 = [3, 2, 2, 1, 1, 0]
    for i in range(6):
        for  j in range(lista_pomocnicza_1[i]):
            town_coords.append([ys[i],xs[2*j+lista_pomocnicza_2[i]]])

    lista_pomocnicza_1 = [6,5,5,4,4,3]
    lista_pomocnicza_2 = [0,1,1,2,2,3]
    for i in range(6):
        for j in range(lista_pomocnicza_1[i]):
            town_coords.append([ys[i+6], xs[2*j + lista_pomocnicza_2[i]]])

    tile_coords = np.array(tile_coords, dtype="float32")
    town_coords = np.array(town_coords, dtype="float32")
    road_coords = np.array(road_coords, dtype="float32")

    tile_img = []
    town_img = []
    road_img = []
    for i, p in enumerate(tile_coords):
        warp = extract_tile(p, img)
        # checkpoint.append(warp.copy())
        tile_img.append(warp.copy())
    for i, p in enumerate(town_coords):
        warp = extract_town(p, img)
        # checkpoint.append(warp.copy())
        town_img.append(warp.copy())

    return tile_img, town_img, road_img, tile_coords, town_coords, road_coords

def findsheep(tile,checkpoint):
    tempcheckpoint=[]
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(rescale(tile, 1.0 / 8.0, anti_aliasing=False)), 0.2, 0.45, 0))))
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1, 1, 1))))
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(tempcheckpoint[len(tempcheckpoint) - 1]), 0.5, 1, 2))))
    #checkpoint.append(erosion_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 1))
    #checkpoint.append(dilation_loop(checkpoint[len(checkpoint) - 1], 2))
    return (threshold(rgb2gray(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1))


def findforest(tile,checkpoint):
    tempcheckpoint = []
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(rescale(tile, 1.0 / 8.0, anti_aliasing=False)), 0.24, 0.45, 0))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint) - 1])
    #tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1, 1, 1))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint) - 1])
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(tempcheckpoint[len(tempcheckpoint) - 1]), 0, 0.8, 2))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint) - 1])
    #checkpoint.append(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.4, 2)))
    #checkpoint.append(erosion_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 1))
    #checkpoint.append(dilation_loop(checkpoint[len(checkpoint) - 1], 8))
    return (threshold(rgb2gray(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1))


def findclay(tile,checkpoint):
    tempcheckpoint = []
    tempcheckpoint.append(hsv2rgb( np.array(filter_colour(rgb2hsv(rescale(tile, 1.0 / 8.0, anti_aliasing=False)), 0, 0.07, 0))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint)-1])
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(tempcheckpoint[len(tempcheckpoint) - 1]), 0.3, 0.8, 1))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint) - 1])
    tempcheckpoint.append(
        hsv2rgb(np.array(filter_colour(rgb2hsv(tempcheckpoint[len(tempcheckpoint) - 1]), 0.0, 0.7, 2))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint) - 1])
    # checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.9, 1))))
    return (threshold(rgb2gray(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1))


def findmountains(tile,checkpoint):
    tempcheckpoint = []
    #tempcheckpoint.append( np.array(filter_colour_reverse(rescale(tile, 1.0 / 8.0, anti_aliasing=False), 0.9, 1, 0.75, 0.95, 0.75, 0.9)))
    #tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0.8, 1, 2)))
    #tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0.8, 1, 0)))
    #tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0.8, 1, 1)))

    tempcheckpoint.append(hsv2rgb(np.array(filter_colour_reverse(rgb2hsv(rescale(tile, 1.0 / 8.0, anti_aliasing=False)), 0.02,0.5 ))))
    # checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.9, 1))))
    return (threshold(rgb2gray(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1))


def findrobber(tile,checkpoint):
    tempcheckpoint = []
    tempcheckpoint.append(util.invert(rescale(tile, 1.0 / 8.0, anti_aliasing=False)))
    tempcheckpoint.append( np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0.9, 1, 0)))
    tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0.9, 1, 1)))
    tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0.9, 1, 2)))
    # checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.9, 1))))
    return (threshold(rgb2gray(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1))

def findtown(tile,checkpoint):
    tempcheckpoint = []
    tempcheckpoint.append(hsv2rgb(np.array(filter_colour_reverse(rgb2hsv(rescale(tile, 1.0 / 8.0, anti_aliasing=False)), 0.06, 0.95))))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint)-1])
    tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0, 0.1, 1)))
    tempcheckpoint.append(np.array(filter_colour(tempcheckpoint[len(tempcheckpoint) - 1], 0, 0.1, 2)))
    #checkpoint.append(tempcheckpoint[len(tempcheckpoint) - 1])
    return (threshold(rgb2gray(tempcheckpoint[len(tempcheckpoint) - 1]), 0.1))

def the_great_tile_finder(tiles):
    sheep=[]
    forest=[]
    clay=[]
    mountains=[]
    robber=[]
    wheat=[]
    temp_list = []
    for i in range(19):
       temp_list.append(pewnosc(findsheep(tiles[i],checkpoint),1))
    i=0
    j=0
    while(i<4):
       index, value= max(enumerate(temp_list), key=operator.itemgetter(1))
       sheep.append(index)
       i = i + 1
       j=j+1
       if(j>19):
           break
       temp_list[index]=0
    temp_list = []
    print(sheep)
    for i in range(19):
       temp_list.append(pewnosc(findforest(tiles[i],checkpoint), 1))
    i=0
    j = 0
    while(i<4):
       index, value= max(enumerate(temp_list), key=operator.itemgetter(1))
       if not index in sheep:
           forest.append(index)
           i=i+1
       j = j + 1
       if (j > 19):
           break
       temp_list[index]=0
    temp_list = []
    print(forest)
    for i in range(19):
        temp_list.append(pewnosc(findclay(tiles[i],checkpoint),1))

    i=0
    j = 0
    while(i<3):
        index, value= max(enumerate(temp_list), key=operator.itemgetter(1))
        if not index in sheep and not index in forest:
            clay.append(index)
            i=i+1
        j = j + 1
        if (j > 19):
            break
        temp_list[index]=0
    temp_list = []
    print(clay)

    for i in range(19):
        temp_list.append(pewnosc(findrobber(tiles[i],checkpoint), 1))

    i = 0
    j = 0
    while (i < 1):
        index, value = max(enumerate(temp_list), key=operator.itemgetter(1))
        if not index in sheep and not index in forest and not index in clay:
            robber.append(index)
            i = i + 1
        j = j + 1
        if (j > 19):
            break
        temp_list[index] = 0
    temp_list = []
    print(robber)

    for i in range(19):
        temp_list.append(pewnosc(findmountains(tiles[i],checkpoint), 1))

    i = 0
    print(temp_list)
    while (i < 3):
        index, value = max(enumerate(temp_list), key=operator.itemgetter(1))
        if not index in sheep and not index in forest and not index in clay and not index in robber:
            mountains.append(index)
            i = i + 1
        temp_list[index] = 0
    temp_list = []
    print(mountains)

    for i in range(0,19):
        if not i in sheep and not i in forest and not i in clay and not i in mountains and i not in robber:
            wheat.append(i)
    print(wheat)

    return sheep,forest,clay,mountains,robber,wheat

def the_great_domki_finder(town_img):
    temp_list = []
    domki=[]
    for town in town_img:
        temp_list.append(pewnosc(findtown(town, checkpoint), 1))
    index, value = max(enumerate(temp_list), key=operator.itemgetter(1))
    while (value>0.05):
        domki.append(index)
        temp_list[index] = 0
        index, value = max(enumerate(temp_list), key=operator.itemgetter(1))

    print(domki)
    return domki



def reverse_warp(warp, checkpoint,coords,dst):
    tmp = coords[:, 0].copy()
    coords[:, 0] = coords[:, 1].copy()
    coords[:, 1] = tmp
    tform3 = tf.ProjectiveTransform()
    tform3.estimate( coords, dst)
    img = tf.warp(warp, tform3, output_shape=(len(warp), len(warp[0])))
    checkpoint.append(img)
    return img

def koloruj_tile(img, lista, kolor, tile_coords):
    dim=75
    for i in lista:
        y = tile_coords[i][0][0]+tile_coords[i][5][0]
        y//=2
        y=int(y)
        x= tile_coords[i][0][1]
        for j in range(-dim,dim):
            for k in range(-dim, dim):
                new_y=int(y+j)
                new_x = int(x+k)
                img[new_y][new_x]=kolor
    return img


def koloruj_town(img, lista, kolor, town_coords):
    dim=50
    for i in lista:
        y = town_coords[i][0]
        x= town_coords[i][1]
        for j in range(-dim,dim):
            for k in range(-dim, dim):
                new_y=int(y+j)
                new_x = int(x+k)
                img[new_y][new_x]=kolor
    return img

def wypelnij(img,data):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                img[i][j] = data[i][j]
    return img

def parallel_check(coords,data,checkpoint,para,exception,przypadek):
    print("parallel_check")
    offset=60
    print(coords[1][1])
    print(coords[3][1])
    print(coords[0][0])
    print(coords[2][0])
    rotated=False
    if coords[3][0]+offset>coords[1][0]>coords[3][0]-offset or coords[0][1]+offset>coords[2][1]>coords[0][1]-offset:
        data = rotate(data, 5)
        checkpoint, contours = background_removal(data.copy())
        checkpoint[1] = erosion_loop(checkpoint[1], 5)
        coords, przypadek,para,exception = kontury_debug(checkpoint[1])
        rotated = True
    return coords, przypadek, data,checkpoint,rotated,para,exception

def zwroc_pokrojone(data, checkpoint):
    data_org = data.copy()
    coords, przypadek,para,exception = kontury_debug(checkpoint[1])
    coords, przypadek, data, checkpoint, rotated,para,exception = parallel_check(coords, data, checkpoint,para,exception,przypadek)
    warp = 0
    dst= 0
    if przypadek == 0:
        warp, dst = transform_if_rectangle(data, coords.copy())
    if przypadek == 1:
        warp, dst = transform_if_not_rect(data, coords.copy())

    tile_img, town_img, road_img, tile_coords, town_coords, road_coords = krojonko(warp, checkpoint)



    sheep, forest, clay, mountains, robber, wheat = the_great_tile_finder(tile_img)
    domki = the_great_domki_finder(town_img)
    print("DEBUG")
    print(sheep)
    print(forest)
    print(clay)
    print(mountains)
    print(robber)
    print(wheat)
    print(domki)
    print("DEBUG")



    # img = data.copy()
    #     # for i in range(len(img)):
    #     #     for j in range(len(img[0])):
    #     #         img[i][j]=[0,0,0]
    #     # img=koloruj_piksel(img,sheep,[1,0,0],tile_coords)
    #     # timestr = time.strftime("%H-%M-%S")
    #     # io.imsave('debug-' + timestr + '.png', img)

    # warp = tf.warp(img, tform3, output_shape=(len(img), len(img[0])))
    # sheep, forest, clay, mountains, robber, wheat
    # checkpoint.append(warp.copy())
    warp = koloruj_tile(warp, sheep, [0, 1, 0], tile_coords)
    # checkpoint.append(warp.copy())
    warp = koloruj_tile(warp, forest, [0, 1, 1], tile_coords)
    # checkpoint.append(warp.copy())
    warp = koloruj_tile(warp, clay, [1, 0, 0], tile_coords)
    # checkpoint.append(warp.copy())
    warp = koloruj_tile(warp, mountains, [0.1, 0.1, 0.1], tile_coords)
    # checkpoint.append(warp.copy())
    warp = koloruj_tile(warp, robber, [1, 1, 0], tile_coords)
    # checkpoint.append(warp.copy())
    warp = koloruj_tile(warp, wheat, [1, 0, 1], tile_coords)
    # checkpoint.append(warp.copy())
    warp = koloruj_town(warp,domki,[1,1,1],town_coords)
    warp = reverse_warp(warp, checkpoint, coords, dst)
    if rotated:
        warp = rotate(warp,-5)
    warp = wypelnij(warp,data_org)
    # timestr = time.strftime("%H-%M-%S")
    # io.imsave('wynik-' + file_string + '-checkpoint-' + str(i) + '-' + timestr + '-warp.png', v)
    # checkpoint.append(warp.copy())
    ploty = wyswietl(checkpoint, nazwapliku)
    ploty[1].plot(coords[:, 1], coords[:, 0], "+r", markersize=15)
    ploty[2].plot(town_coords[:, 1], town_coords[:, 0], "+r", markersize=15)
    return warp, tile_img, town_img


if __name__ == '__main__':

    start_time = time.time()
    # for file in range(21, 30):
    for file in [21,22]:
        # for file in range(31, 43):
        nazwapliku = str(file) + ".jpg"
        print(nazwapliku)
        data = io.imread(nazwapliku)
        data = img_as_float(data)

        checkpoint, contours = background_removal(data.copy())

        checkpoint[1] = erosion_loop(checkpoint[1], 5)

        warp, tile_img,town_img = zwroc_pokrojone(data, checkpoint)

        timestr = time.strftime("%H-%M-%S")
        file_string = str(file)
        if file < 10:
           file_string = '0' + file_string
        io.imsave('wynik-' + file_string + '-checkpoint-' + timestr + '-warp.png', warp)
        #io.imsave('wynik-' + file_string + '-' + timestr + '-warp.png', warp)
        #savefig('wynik-' + file_string + '-' + timestr + '.png')
        #for i, v in enumerate(tile_img):
        #    io.imsave('wynik-' + file_string + '-tile-' + str(i) + '-' + timestr + '-warp.png', v)
        # for i, v in enumerate(checkpoint):
        #     io.imsave('wynik-' + file_string + '-checkpoint-' + str(i) + '-' + timestr + '-warp.png', v)
        # savefig('wynik-' + file_string + '-' + timestr + '.png')
        #io.show()

        # rows = 5
        # columns = 5
        # fig = plt.figure(figsize=(10*columns, rows * 10))
        # l1=[1,0,0,0,1]
        # l2=[3,4,5,4,3]
        # k=0
        # for i in range(rows):
        #     for j in range(l2[i]):
        #         ax = fig.add_subplot(rows, columns, 5*i+l1[i]+j+1)
        #         print(k)
        #         io.imshow(tile_img[k])
        # savefig('wynik-' + file_string + '-' + timestr + '2.png')
        # io.show()

    print("czas wykonywania:", time.time() - start_time)

    # checkpoint.append(outer_removal(checkpoint[1]))
    # checkpoint.append(leave_only_island(checkpoint[0], checkpoint[2]))

    # checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.5,1))))

    # findforest(checkpoint,3)
    # contours1 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    # findsheep(checkpoint,3)
    # contours2 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    # findclay(checkpoint,3)
    # contours3 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    # findmountains(checkpoint, 3)
    # contours4 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    # findwheat(checkpoint, 3)
    # contours5 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.14)
    # ploty = wyswietl(checkpoint)
    # contours1 = sort_contourow(contours1)
    # contours1 = top_contoury(contours1,4)
    # kontury_do_srodkow(contours1, ploty[0], 0.25)
    # contours2 = sort_contourow(contours2)
    # contours2 = top_contoury(contours2, 4)
    # kontury_do_srodkow(contours2, ploty[0], 0.4)
    # contours3 = sort_contourow(contours3)
    # contours3 = top_contoury(contours3, 3)
    # kontury_do_srodkow(contours3, ploty[0], 0)
    # contours4 = sort_contourow(contours4)
    # contours4 = top_contoury(contours4, 3)
    # kontury_do_srodkow(contours4, ploty[0], 0.8)
    # contours5 = sort_contourow(contours5)
    # contours5 = top_contoury(contours5, 4)
    # kontury_do_srodkow(contours5, ploty[0], 0.7)

    # coords = measure.approximate_polygon(contours[najwiekszy_contour(contours)], tolerance=2)
    # coords=usuwanko_punktow(coords)

    io.show()
    # print(time.time() - start_time)
