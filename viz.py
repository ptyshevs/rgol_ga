import matplotlib.pyplot as plt

def show_field(field, reshape=True):
    """
    Plot field as 2D raster image
    """
    if reshape:
        field = field.reshape((20, 20))
    plt.imshow(field, cmap=plt.cm.Greys_r)
    plt.xticks([], []);
    plt.yticks([], []);
    
