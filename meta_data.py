import numpy as np

# =========================================================
# PATH CONFIG (EDIT THIS )
# =========================================================
DATA_PATH = "data"  # change to Drive path 

# =========================================================
# SENSOR METADATA
# =========================================================

vnir_meta = {
    "wavelength": np.load(f"{DATA_PATH}/VNIR_wavelength.npy")
}

swir_meta = {
    "wavelength": np.load(f"{DATA_PATH}/SWIR_wavelength.npy")
}
