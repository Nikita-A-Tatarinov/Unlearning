import logging
import os

LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

ROOT_DIR: str = '.' + os.sep

SAFERLHF_DIR: str = os.path.join(ROOT_DIR, 'SafeRLHF')
