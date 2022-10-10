import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if os.name == "posix":
    data_dir = Path("/home/alex/Data/THzConductivity/MarielenaData")
else:
    data_dir = Path("E:\measurementdata\THz Conductivity")

