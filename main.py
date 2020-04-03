from skimage import io, measure
from matplotlib import pylab as plt
from skimage.feature import corner_harris, corner_subpix, corner_peaks
io.use_plugin('matplotlib')

def display(checkpoint, nazwa, contour):
    rows = len(checkpoint)
    columns = 1
    fig = plt.figure(figsize=(5, rows * 5))
    ploty = []
    for i in range(rows):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.plot(contour[i][:, 1], contour[i][:, 0], 'ro', markersize=5, linewidth=2)
        ax.set_title(nazwa)
        ploty.append(ax)
        io.imshow(checkpoint[i])

    return ploty



if __name__ == "__main__":
    checkpoint = []
    contours = []
    nazwapliku = "set0/0.png"
    print(nazwapliku)
    data = io.imread(nazwapliku)
    checkpoint.append(data)
    contours.append(corner_peaks(corner_harris(data), min_distance=5))
    checkpoint.append(data)
    contours.append(corner_subpix(data, contours[-1], window_size=13))
    display(checkpoint, '', contours)
    io.show()
