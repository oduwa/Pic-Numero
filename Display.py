from scipy import misc
import matplotlib.pyplot as plt

def show_image(image, title="", isGray=True):
    fig, ax = plt.subplots()
    if(isGray == True):
        plt.gray();
    ax.imshow(image, interpolation='nearest')
    ax.set_title(title)
    plt.show()
