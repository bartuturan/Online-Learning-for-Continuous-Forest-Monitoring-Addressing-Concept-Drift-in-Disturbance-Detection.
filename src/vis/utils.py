import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap_clouds = ListedColormap([
    '#87CEEB',  # free_sky
    '#FFFFFF',  # cloud
    '#696969',  # cloud_shadows
    '#B0E0E6',  # snow
    '#FFA500',  # masked_other_reasons
    '#000000'   # no_data
])
cloud_categories = np.asarray([0, 1, 2, 3, 4, 5])
norm_clouds = BoundaryNorm(boundaries=np.concatenate(([cloud_categories[0] - 0.5], (cloud_categories[:-1] + cloud_categories[1:]) / 2, [cloud_categories[-1] + 0.5])), ncolors=cmap_clouds.N)


cmap_scl = ListedColormap([
    '#000000',  # no_data
    '#FF00FF',  # saturated_or_defective
    '#2F4F4F',  # dark_area_pixels
    '#696969',  # cloud_shadows
    '#006400',  # vegetation
    '#D2B48C',  # bare_soils
    '#0000FF',  # water
    '#C0C0C0',  # clouds_low_probability_or_unclassified
    '#FFFFFF',  # clouds_medium_probability
    '#FFFF99',  # clouds_high_probability
    '#E0FFFF',  # cirrus
    '#87CEEB'   # snow_or_ice
])
scl_categories= np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
norm_scl = BoundaryNorm(boundaries=np.concatenate(([scl_categories[0] - 0.5], (scl_categories[:-1] + scl_categories[1:]) / 2, [scl_categories[-1] + 0.5])), ncolors=cmap_scl.N)

cmap_forest = ListedColormap([
    '#A9A9A9',  # non-forest
    '#008000',  # forest
])
forest_categories = np.asarray([0, 1])
norm_forest = BoundaryNorm(boundaries=np.concatenate(([forest_categories[0] - 0.5], (forest_categories[:-1] + forest_categories[1:]) / 2, [forest_categories[-1] + 0.5])), ncolors=cmap_forest.N)

cmap_disturbances = ListedColormap([
    '#A9A9A9',  # no_disturbance
    '#FF0000',  # disturbance
    '#000000'   # no_data
])
disturbances_categories = np.asarray([0, 1, 255])
norm_disturbances = BoundaryNorm(boundaries=np.concatenate(([disturbances_categories[0] - 0.5], (disturbances_categories[:-1] + disturbances_categories[1:]) / 2, [disturbances_categories[-1] + 0.5])), ncolors=cmap_disturbances.N)


cmap_disturbanceag = ListedColormap([
    '#8B4513',  # wind/bark_beetle_complex
    '#FF4500',  # fire
    '#228B22',  # harvest
    '#800080',  # mixed_agents
    '#000000'   # no_data
])
disturbanceag_categories = np.asarray([0, 1, 2, 3, 255])
norm_disturbanceag = BoundaryNorm(boundaries=np.concatenate(([disturbanceag_categories[0] - 0.5], (disturbanceag_categories[:-1] + disturbanceag_categories[1:]) / 2, [disturbanceag_categories[-1] + 0.5])), ncolors=cmap_disturbanceag.N)
