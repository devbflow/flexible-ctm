import torchvision.datasets.utils as utils
import os
from pathlib import Path

CWD = os.getcwd()
mIn_PATH = Path(CWD+"/miniImagenet")
try:
    os.mkdir(mIn_PATH)
    utils.download_file_from_google_drive("1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk", mIn_PATH, "miniImagenet.zip")
except FileExistsError:
    pass
