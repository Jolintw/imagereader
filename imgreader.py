import matplotlib.pyplot as plt
import numpy as np

class Imgreader:
    def __init__(self, filename):
        self.im = plt.imread(filename)

    def makegrid(self):
        """
        xind_1d: int(nx), yind_1d: int(ny)\n
        xratio_1d: float(nx), yratio: float(ny) from 0 to 1\n
        xind, yind: int(ny, nx)\n
        xratio, yratio: float(ny, nx) from 0 to 1
        """
        xlen = self.im.shape[1]
        ylen = self.im.shape[0]
        self.xind_1d, self.yind_1d = np.arange(xlen, dtype=int), np.arange(ylen, dtype=int)
        self.xratio_1d, self.yratio_1d = self.xind_1d / xlen, self.yind_1d / ylen
        self.xind, self.yind = np.meshgrid(self.xind_1d, self.yind_1d)
        self.xratio, self.yratio = np.meshgrid(self.xratio_1d, self.yratio_1d)

    def cut_img(self, x0, x1, y0, y1):
        shape = self.im.shape
        xind_lim = [int(np.round(shape[1]*x0)), int(np.round(shape[1]*x1))]
        yind_lim = [int(np.round(shape[0]*y0)), int(np.round(shape[0]*y1))]
        self.origin_im = self.im.copy()
        self.im = self.im[yind_lim[0]:yind_lim[1]+1, xind_lim[0]:xind_lim[1]+1, :]
        return self.im
    
    def yflip(self):
        self.im = self.im[::-1, ...]
        return self.im

    def mask_by_color(self, color, tolerance = 0.01):
        im = self.im
        try:
            iter(color[0])
        except:
            color = (color, )
        
        mask = np.zeros(im.shape[:2], dtype=bool)
        for c in color:
            colorlen = len(c)
            # print(np.mean(np.abs(im[..., :colorlen] - np.array(c))))
            mask[np.mean(np.abs(im[..., :colorlen] - np.array(c)), axis=2) <= tolerance] = True
        return mask
    
    def value_by_color(self, color, value, tolerance = 0.01, fill_value = 0):
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
            value_of_img[np.mean(np.abs(im[..., :colorlen] - np.array(c)), axis=2) <= tolerance] = v
        return value_of_img
    
    def set_original_limit(self, xlim, ylim):
        """
        xlim = (xmin, xmax)\n
        ylim = (ymin, ymax)
        """
        if xlim is None:
            xlim = self.xlim
        else:
            self.xlim = xlim
        if ylim is None:
            ylim = self.ylim
        else:
            self.ylim = ylim
        return self.xlim, self.ylim

    def ratioxy_to_originalxy(self, ratiox, ratioy, original_xlim = None, original_ylim = None):
        xlim, ylim = self.set_original_limit(original_xlim, original_ylim)
        originx = ratiox * (xlim[1] - xlim[0]) + xlim[0]
        originy = ratioy * (ylim[1] - ylim[0]) + ylim[0]
        return originx, originy
    
    def originalxy_to_ratioxy(self, originx, originy, original_xlim = None, original_ylim = None):
        xlim, ylim = self.set_original_limit(original_xlim, original_ylim)
        ratiox = (originx - xlim[0]) / (xlim[1] - xlim[0])
        ratioy = (originy - ylim[0]) / (ylim[1] - ylim[0])
        return ratiox, ratioy

    def show(self):
        plt.imshow(self.im)
        plt.show()