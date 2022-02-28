# Paths of data files
SRC_FOLDER_PATH = './dataset/source'
LABEL_CSV_PATH = './dataset/labels/label.csv'
INPUT_DATA_PATH = './dataset/input_data'
H5FILE = f'{INPUT_DATA_PATH}/dataset.h5'

# Dataset parameters
TEST_SPLIT_SIZE = 0.2
EPOCH_NUM = 1000
BATCH_SIZE = 5

# Settings for training
MODEL = "3DCONV"
MODEL_NAME = f'LIPREADING_{MODEL}'
TRAINING_FROM_SCRATCH = True
PATH_CURVE = f'./training_results/Curve/{MODEL_NAME}.png'
PATH_WEIGHTS = f'./training_results/Weights/{MODEL_NAME}'
