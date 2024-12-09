import matplotlib
import numpy as np

red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0, 50)[1:]])
green = np.concatenate([np.linspace(0.5, 1, 50), np.linspace(1, 0, 50)[1:]])
blue = np.zeros(99)

rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), blue.reshape(-1, 1)], axis=1)

color_map_epf = [
    [0, 0.5120, 0],
    [0.0244, 0.5569, 0],
    [0.0488, 0.5987, 0],
    [0.0732, 0.6375, 0],
    [0.0976, 0.6733, 0],
    [0.1220, 0.7062, 0],
    [0.1463, 0.7364, 0],
    [0.1707, 0.7638, 0],
    [0.1951, 0.7888, 0],
    [0.2195, 0.8114, 0],
    [0.2439, 0.8318, 0],
    [0.2683, 0.8501, 0],
    [0.2927, 0.8667, 0],
    [0.3171, 0.8815, 0],
    [0.3415, 0.8947, 0],
    [0.3659, 0.9066, 0],
    [0.3902, 0.9172, 0],
    [0.4146, 0.9266, 0],
    [0.4390, 0.9350, 0],
    [0.4634, 0.9424, 0],
    [0.4878, 0.9490, 0],
    [0.5122, 0.9549, 0],
    [0.5366, 0.9601, 0],
    [0.5610, 0.9648, 0],
    [0.5854, 0.9689, 0],
    [0.6098, 0.9725, 0],
    [0.6341, 0.9757, 0],
    [0.6585, 0.9785, 0],
    [0.6829, 0.9810, 0],
    [0.7073, 0.9833, 0],
    [0.7317, 0.9852, 0],
    [0.7561, 0.9870, 0],
    [0.7805, 0.9885, 0],
    [0.8049, 0.9898, 0],
    [0.8293, 0.9910, 0],
    [0.8537, 0.9921, 0],
    [0.8780, 0.9930, 0],
    [0.9024, 0.9938, 0],
    [0.9268, 0.9946, 0],
    [0.9512, 0.9952, 0],
    [0.9756, 0.9958, 0],
    [1.0000, 0.9963, 0],
    [1, 1, 0],
    [0.9963, 0, 0],
    [0.9958, 0, 0],
    [0.9952, 0, 0],
    [0.9946, 0, 0],
    [0.9938, 0, 0],
    [0.9930, 0, 0],
    [0.9921, 0, 0],
    [0.9910, 0, 0],
    [0.9898, 0, 0],
    [0.9885, 0, 0],
    [0.9870, 0, 0],
    [0.9852, 0, 0],
    [0.9833, 0, 0],
    [0.9810, 0, 0],
    [0.9785, 0, 0],
    [0.9757, 0, 0],
    [0.9725, 0, 0],
    [0.9689, 0, 0],
    [0.9648, 0, 0],
    [0.9601, 0, 0],
    [0.9549, 0, 0],
    [0.9490, 0, 0],
    [0.9424, 0, 0],
    [0.9350, 0, 0],
    [0.9266, 0, 0],
    [0.9172, 0, 0],
    [0.9066, 0, 0],
    [0.8947, 0, 0],
    [0.8815, 0, 0],
    [0.8667, 0, 0],
    [0.8501, 0, 0],
    [0.8318, 0, 0],
    [0.8114, 0, 0],
    [0.7888, 0, 0],
    [0.7638, 0, 0],
    [0.7364, 0, 0],
    [0.7062, 0, 0],
    [0.6733, 0, 0],
    [0.6375, 0, 0],
    [0.5987, 0, 0],
    [0.5569, 0, 0],
    [0.5120, 0, 0],
    [0, 0, 0],
]

# Defining color map
color_map_epf = matplotlib.colors.ListedColormap(color_map_epf)
