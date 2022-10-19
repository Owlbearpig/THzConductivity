import os
from pathlib import Path
from scipy.constants import c as c0
from numpy import pi

um = 10**-6
THz = 10 ** 12

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if os.name == "posix":
    data_dir = Path("/home/alex/Data/THzConductivity/MarielenaData")
    teralyzer_result_dir = Path(r"/home/alex/MEGA/AG/Projects/THz Conductivity/Results/Teralyzer")
else:
    data_dir = Path("E:\measurementdata\THz Conductivity")
    teralyzer_result_dir = None

