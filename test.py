from skimage import io, measure, feature, transform
from matplotlib import pylab as plt
from scipy import ndimage as ndi
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import numpy as np
from typing import List
import ramda as R
import math
import copy

io.use_plugin('matplotlib')


def break_edges(edges: np.ndarray):
    left = np.zeros_like(np.arange(edges.shape[0]))
    right = np.zeros_like(np.arange(edges.shape[0]))
    top = np.zeros_like(np.arange(edges.shape[1]))
    bottom = np.zeros_like(np.arange(edges.shape[1]))
    print(edges.shape, left.shape, right.shape)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j]:
                left[i] = j
                break
        else:
            left[i] = -1
            right[i] = -1
        if right[i] == 0:
            for j in range(edges.shape[1] - 1, -1, -1):
                if edges[i][j]:
                    right[i] = j
                    break

    for j in range(edges.shape[1]):
        for i in range(edges.shape[0]):
            if edges[i][j]:
                top[j] = i
                break
        else:
            top[j] = -1
            bottom[j] = -1
        if bottom[j] == 0:
            for i in range(edges.shape[0] - 1, -1, -1):
                if edges[i][j]:
                    bottom[j] = i
                    break
    top_left = []
    top_right = []
    bot_left = []
    bot_right = []
    for i in range(edges.shape[0]):
        if left[i] >= 0:
            l = left[i]
            if top[l] == i:
                top_left.append([i, l])
            if bottom[l] == i:
                bot_left.append([i, l])
        if right[i] >= 0:
            l = right[i]
            if top[l] == i:
                top_right.append([i, l])
            if bottom[l] == i:
                bot_right.append([i, l])
    return [top_left, top_right, bot_right, bot_left]


def edges_to_functions(edges):
    result = []
    for edge in edges:
        # z założenia są monotoniczne
        min_e = R.reduce(R.min_by(lambda x: x[0]), [1e9, 0], edge)
        max_e = R.reduce(R.max_by(lambda x: x[0]), [0, 0], edge)
        a = (max_e[1] - min_e[1]) / (max_e[0] - min_e[0])
        b = min_e[1] - a * min_e[0]
        result.append([a, b])
    return result


def blad_srednikwadratowy(edges, functions):
    result = []
    for edge, wspolczynniki in zip(edges, functions):
        f = lambda x: wspolczynniki[0] * x + wspolczynniki[1]
        errors = R.map(lambda x: (f(x[0]) - x[1]) ** 2)(edge[1:-1])
        result.append(0 if len(errors)<1 else sum(errors) / len(errors))
    return result


def angles_between_functions(functions):
    for i, [a1, b1] in enumerate(functions):
        print(i)

        for j, [a2, b2] in enumerate(functions):
            if i == j: continue
            tan = (a1 - a2) / (1 + a1 * a2)
            print(j, tan, math.degrees(math.atan(tan)))


def angles_with_neighbouring_functions(functions):
    results = []
    for i, [a1, b1] in enumerate(functions):
        print(i)
        r = []
        for j in [(i - 1) % 4, (i + 1) % 4]:
            a2 = functions[j][0]
            tan = (a1 - a2) / (1 + a1 * a2)
            r.append(abs(math.degrees(math.atan(tan))))
        results.append(r)
    return results


def list_to_display(shape, my_list):
    new_boi = np.full(shape, False)
    for i in my_list:
        new_boi[i[0]][i[1]] = True
    return new_boi


if __name__ == "__main__":
    checkpoint = []
    contours = []
    how_many_in_folder = [6, 20, 20, 20, 20, 200, 20, 100]
    for set_nr in range(9):
        for img_nr in range(how_many_in_folder[set_nr]):
            # nazwa_pliku = "set{}/{}.png".format(set_nr, img_nr)
            # Problemy z: (7,15)
            nazwa_pliku = "set{}/{}.png".format(set_nr, img_nr)
            print(nazwa_pliku)
            im = io.imread(nazwa_pliku)
            # im2 = copy.deepcopy(im)
            edges1 = feature.canny(im)
            edges = break_edges(edges1)
            functions = edges_to_functions(edges)
            errors = blad_srednikwadratowy(edges, functions)
            print(errors)
            # angles_between_functions(functions)

            neighbour_angles = angles_with_neighbouring_functions(functions)
            print(neighbour_angles)
            
            neighbour_angles_sum =R.map(sum, neighbour_angles)
            print(neighbour_angles_sum)

            max_index = lambda my_list: R.reduce(lambda acc, x: acc if acc[1] >= x[1] else x, [0, 0], R.zip(range(len(my_list)), my_list))
            e = max_index(errors)
            print(e)
            a = max_index(neighbour_angles_sum)
            print(a)

            #wg błędów
            index_podstawy = (e[0] + 2) % 4
            print('wg błędów', index_podstawy, functions[index_podstawy])
            [a1, _] = functions[index_podstawy]

            #wg kątów
            index_podstawy2 = a[0]
            print('wg kątów',index_podstawy2, functions[index_podstawy2])
            [a2, _] = functions[index_podstawy2]

            im2 = transform.rotate(im, 90 - math.degrees(math.atan(a1)))
            im3 = transform.rotate(im, 90 - math.degrees(math.atan(a2)))
            memes = []
            for i in edges:
                # print(i)
                memes.append(list_to_display(im.shape, i))

            row_count = 2 + len(memes) + 2
            fig, axes = plt.subplots(nrows=row_count, ncols=1, figsize=(3, 3 * row_count),
                                     sharex=True, sharey=True)

            axes[0].imshow(im, cmap=plt.cm.gray)
            axes[0].axis('off')
            # axes[0].set_title('noisy image', fontsize=20)

            axes[1].imshow(edges1, cmap=plt.cm.gray)
            axes[1].axis('off')

            for i, m in enumerate(memes):
                axes[2 + i].imshow(m, cmap=plt.cm.gray)
                axes[2 + i].axis('off')

            axes[6].imshow(im2, cmap=plt.cm.gray)
            axes[6].axis('off')
            axes[7].imshow(im3, cmap=plt.cm.gray)
            axes[7].axis('off')
            fig.tight_layout()

            plt.show()
            # break
        # break
        if set_nr>=0:
            break