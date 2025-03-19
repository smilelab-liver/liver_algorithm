import numpy as np


def get_class_color(class_idx):
    """
    bile duct: (255, 0, 0)
    portal vein: (0, 0, 255)
    central vein: (0, 255, 0)
    """
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    return colors[class_idx]

def get_randoom_color():
    return np.random.randint(0, 256, size=3, dtype=np.uint8)