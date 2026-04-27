import matplotlib.pyplot as plt

def add_colorbar(g, cube_var, norm, fontsize=8):
    try:
        g.add_colorbar(label='Category')
    except:
        g = DummyG(g)
        g.add_colorbar(label='Category')

    # Set ticks (centered for discrete bins)
    g.cbar.set_ticks((norm.boundaries[:-1] + norm.boundaries[1:]) / 2)

    # Set tick labels manually
    g.cbar.ax.set_yticklabels(cube_var.attrs["flag_meanings"])
    g.cbar.ax.tick_params(labelsize=fontsize)  # optional: smaller font


class DummyG:
    def __init__(self, im):
        self.im = im
        self.ax = im.axes
        self.cbar = None  # will be set below

    def add_colorbar(self, **kwargs):
        self.cbar = plt.colorbar(self.im, ax=self.ax, **kwargs)