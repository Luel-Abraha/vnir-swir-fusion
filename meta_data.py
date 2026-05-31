import numpy as np

# =========================================================
# PATH CONFIG (EDIT ONLY THIS IF NEEDED)
# =========================================================
DATA_PATH = "data"  # change to Drive path if needed

# =========================================================
# SENSOR METADATA
# =========================================================

vnir_meta = {
    "wavelength": np.load(f"{DATA_PATH}/VNIR_wavelength.npy")
}

swir_meta = {
    "wavelength": np.load(f"{DATA_PATH}/SWIR_wavelength.npy")
}
