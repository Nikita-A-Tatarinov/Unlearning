import os

from config import SAFERLHF_DIR

LOG_DIR: str = os.path.join(SAFERLHF_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)
CLS_LOG_FILE_NAME: str = 'classification'
CLS_LOG_FILE: str = os.path.join(
    LOG_DIR, f'{CLS_LOG_FILE_NAME}.txt')
TRANSFORMATION_LOG_FILE_NAME: str = 'transformation'
TRANSFORMATION_LOG_FILE: str = os.path.join(
    LOG_DIR, f'{TRANSFORMATION_LOG_FILE_NAME}.txt')


DATA_DIR: str = os.path.join(SAFERLHF_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
ORIGINAL_DATA_FILE_NAME: str = 'original'
ORIGINAL_DATA_FILE: str = os.path.join(
    DATA_DIR, f'{ORIGINAL_DATA_FILE_NAME}.csv')

CLS_DATA_DIR: str = os.path.join(DATA_DIR, 'classification')
os.makedirs(CLS_DATA_DIR, exist_ok=True)
CLS_DATA_STATS_FILE_NAME: str = 'metadata'
CLS_DATA_STATS_FILE: str = os.path.join(
    CLS_DATA_DIR, f'{CLS_DATA_STATS_FILE_NAME}.csv')

CLS_SAFE_DATA_FILE_NAME: str = 'safe'
CLS_SAFE_DATA_FILE: str = os.path.join(
    CLS_DATA_DIR, f'{CLS_SAFE_DATA_FILE_NAME}.csv')

CLS_TRAIN_TEST_DATA_DIR: str = os.path.join(CLS_DATA_DIR, 'train_test')
os.makedirs(CLS_TRAIN_TEST_DATA_DIR, exist_ok=True)

TRANSFORMATION_DATA_DIR: str = os.path.join(DATA_DIR, 'transformation')
os.makedirs(TRANSFORMATION_DATA_DIR, exist_ok=True)
TRANSFORMATION_DATA_STATS_FILE_NAME: str = 'metadata'
TRANSFORMATION_DATA_STATS_FILE: str = os.path.join(
    TRANSFORMATION_DATA_DIR, f'{TRANSFORMATION_DATA_STATS_FILE_NAME}.csv')

TRANSFORMATION_TRAIN_TEST_DATA_DIR: str = os.path.join(
    TRANSFORMATION_DATA_DIR, 'train_test')
os.makedirs(TRANSFORMATION_TRAIN_TEST_DATA_DIR, exist_ok=True)

MODELS_DIR: str = os.path.join(SAFERLHF_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CLS_MODELS_DIR: str = os.path.join(MODELS_DIR, 'classification')
os.makedirs(CLS_MODELS_DIR, exist_ok=True)
CLS_MODELS_PLOTS_DIR: str = os.path.join(CLS_MODELS_DIR, 'plots')
os.makedirs(CLS_MODELS_PLOTS_DIR, exist_ok=True)

TRANSFORMATION_MODELS_DIR: str = os.path.join(MODELS_DIR, 'transformation')
os.makedirs(TRANSFORMATION_MODELS_DIR, exist_ok=True)
TRANSFORMATION_MODELS_PLOTS_DIR: str = os.path.join(
    TRANSFORMATION_MODELS_DIR, 'plots')
os.makedirs(TRANSFORMATION_MODELS_PLOTS_DIR, exist_ok=True)
