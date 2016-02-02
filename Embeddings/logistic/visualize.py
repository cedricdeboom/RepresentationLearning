__author__ = 'cedricdeboom'


import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def show_matrix(matrix, output='output.pdf'):
    #maximum_value = np.max(matrix)
    #minimum_value = np.min(matrix)

    #matrix = (matrix - minimum_value) / (maximum_value - minimum_value)

    plt.pcolor(matrix, cmap=cm.Greys)
    plt.colorbar()
    plt.savefig(output)
    plt.close()