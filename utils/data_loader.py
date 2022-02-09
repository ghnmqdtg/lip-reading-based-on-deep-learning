import os
import sys
import numpy as np
import pandas as pd

# Adding the parent directory to the sys.path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
import config

def load_label_csv():
    label = pd.read_csv(config.LABEL_CSV_PATH)
    label = np.array(label)
    return label

if __name__ == '__main__':
    load_label_csv()