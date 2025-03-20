import matplotlib.pyplot as plt
import numpy as np

class Imgreader:
    def __init__(self, filename):
        self.im = plt.imread(filename)

    def cut_img(self, x0, x1, y0, y1):
        shape = self.im.shape
        xind_lim = [int(np.round(shape[1]*x0)), int(np.round(shape[1]*x1))]
        yind_lim = [int(np.round(shape[0]*y0)), int(np.round(shape[0]*y1))]
        self.im = self.im[yind_lim[0]:yind_lim[1]+1, xind_lim[0]:xind_lim[1]+1, :]
        return self.im
    
    def yflip(self):
        self.im = self.im[::-1, ...]
        return self.im

    def color_to_mask(self, color, tolerance = 0.01):
        im = self.im
        try:
            iter(color[0])
        except:
            color = (color, )
        colorlen = len(color[0])
        mask = np.zeros(im.shape[:2], dtype=bool)
        for c in color:
            mask[np.mean(np.abs(im[..., :colorlen] - np.array(c))) <= tolerance] = True
        return mask
    
    def color_to_value(self, color, value, tolerance = 0.01, fill_value = 0):
        im = self.im
        try:
            iter(color[0])
        except:
            color = (color, )
        try:
            iter(value)
        except:
            value = (value, )
        colorlen = len(color[0])
        value_of_img = np.zeros(im.shape[:2]) + fill_value
        for c, v in zip(color, value):
            value_of_img[np.mean(np.abs(im[..., :colorlen] - np.array(c))) <= tolerance] = v
        return value_of_img

    def show(self):
        plt.imshow(self.im)
        plt.show()