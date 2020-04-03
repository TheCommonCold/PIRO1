from skimage import io
from matplotlib import pylab as plt
io.use_plugin('matplotlib')

def display(checkpoint, nazwa):
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


if __name__ == "__main__":
    checkpoint = []
    nazwapliku = "set0/0.png"
    print(nazwapliku)
    data = io.imread(nazwapliku)
    checkpoint.append(data)
    display(checkpoint,'')
    io.show()
