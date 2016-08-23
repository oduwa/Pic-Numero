from scipy import misc
import matplotlib.pyplot as plt

def show_image(image, title="", isGray=True):
    fig, ax = plt.subplots()
    if(isGray == True):
        plt.gray();
    ax.imshow(image, interpolation='nearest')
    ax.set_title(title)
    plt.show()

def save_image(filename, image, title="", isGray=True):
    fig, ax = plt.subplots()
    plt.axis('off')
    if(isGray == True):
        plt.gray();
    ax.imshow(image, interpolation='nearest')
    ax.set_title(title)
    plt.savefig(filename)
    plt.close(fig)
