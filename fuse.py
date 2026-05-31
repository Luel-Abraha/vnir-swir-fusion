import numpy as np
import meta_data
import matplotlib.pyplot as plt

# =========================================================
# DATA PATH
# =========================================================

DATA_PATH = "data/aligned_cubes" 

# =========================================================
# LOAD DATA
# =========================================================

vnir_cal_crop = np.load(
    f"{DATA_PATH}/VNIR_cropped_cube.npy",
    mmap_mode='r'
)

SWIR_aligned_cube = np.load(
    f"{DATA_PATH}/SWIR_aligned_cube.npy",
    mmap_mode='r'
)

# =========================================================
# NORMALIZATION
# =========================================================

def normalize_channel(band):
    low, high = np.percentile(band, (1, 99))
    band = np.clip(band, low, high)
    band = (band - low) / (high - low + 1e-8)
    return band

# =========================================================
# SYNTHETIC RGB
# =========================================================

def VNIR_synthetic_RGB(cube):
    R, G, B = 82, 47, 20

    r = normalize_channel(cube[:, :, R])
    g = normalize_channel(cube[:, :, G])
    b = normalize_channel(cube[:, :, B])

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def SWIR_synthetic_RGB(cube):
    R2, G2, B2 = 207, 112, 67

    r2 = normalize_channel(cube[:, :, R2])
    g2 = normalize_channel(cube[:, :, G2])
    b2 = normalize_channel(cube[:, :, B2])

    rgb2 = np.stack([r2, g2, b2], axis=-1)
    return rgb2

# =========================================================
# OVERLAP BAND DETECTION
# =========================================================

def get_overlap_bands(vnir_waves,
                      swir_waves,
                      overlap_min=951,
                      overlap_max=1000):

    vnir_idx = np.where(
        (vnir_waves >= overlap_min) &
        (vnir_waves <= overlap_max)
    )[0]

    swir_idx = np.where(
        (swir_waves >= overlap_min) &
        (swir_waves <= overlap_max)
    )[0]

    return vnir_idx, swir_idx

# =========================================================
# RESAMPLING
# =========================================================

def resample_swir_to_vnir(swir_overlap,
                          swir_waves_overlap,
                          vnir_waves_overlap):

    H, W, B = swir_overlap.shape
    swir_flat = swir_overlap.reshape(-1, B)

    N = swir_flat.shape[0]
    T = len(vnir_waves_overlap)

    out = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        out[i] = np.interp(
            vnir_waves_overlap,
            swir_waves_overlap,
            swir_flat[i]
        )

    return out.reshape(H, W, T)

# =========================================================
# MULTIPLICATIVE CORRECTION
# =========================================================

def multiplicative_correction(vnir_overlap, swir_overlap):

    B = vnir_overlap.shape[2]
    swir_corrected = np.zeros_like(swir_overlap)

    for b in range(B):
        vnir_mean = np.mean(vnir_overlap[:, :, b])
        swir_mean = np.mean(swir_overlap[:, :, b])
        gain = vnir_mean / (swir_mean + 1e-8)

        swir_corrected[:, :, b] = swir_overlap[:, :, b] * gain

    return swir_corrected

# =========================================================
# VALIDATION (MAD)
# =========================================================

def validate_fusion(vnir_overlap, swir_original, swir_corrected):

    err_before = np.mean(np.abs(vnir_overlap - swir_original))
    err_after = np.mean(np.abs(vnir_overlap - swir_corrected))

    improvement = (err_before - err_after) / (err_before + 1e-12) * 100

    return {
        "MAD_before": err_before,
        "MAD_after": err_after,
        "improvement_%": improvement
    }

# =========================================================
# MAIN FUSION
# =========================================================

def fuse_vnir_swir(vnir_cube,
                   swir_cube,
                   vnir_waves,
                   swir_waves,
                   vnir_overlap_idx,
                   swir_overlap_idx):

    vnir_cube = vnir_cube.astype(np.float32)
    swir_cube = swir_cube.astype(np.float32)

    # masks
    vnir_mask = np.ones(vnir_cube.shape[2], dtype=bool)
    vnir_mask[vnir_overlap_idx] = False

    swir_mask = np.ones(swir_cube.shape[2], dtype=bool)
    swir_mask[swir_overlap_idx] = False

    vnir_non_overlap = vnir_cube[:, :, vnir_mask]
    swir_non_overlap = swir_cube[:, :, swir_mask]

    # overlap
    vnir_overlap = vnir_cube[:, :, vnir_overlap_idx]
    swir_overlap = swir_cube[:, :, swir_overlap_idx]

    vnir_waves_overlap = vnir_waves[vnir_overlap_idx]
    swir_waves_overlap = swir_waves[swir_overlap_idx]

    # resample
    swir_resampled = resample_swir_to_vnir(
        swir_overlap,
        swir_waves_overlap,
        vnir_waves_overlap
    )

    # correct
    swir_corrected = multiplicative_correction(
        vnir_overlap,
        swir_resampled
    )

    # validate
    results = validate_fusion(
        vnir_overlap,
        swir_resampled,
        swir_corrected
    )

    print(f"MAD before: {results['MAD_before']:.6f}")
    print(f"MAD after: {results['MAD_after']:.6f}")
    print(f"Improvement: {results['improvement_%']:.1f}%")

    # fuse
    fused_overlap = (vnir_overlap + swir_corrected) / 2

    fused_cube = np.concatenate(
        [vnir_non_overlap, fused_overlap, swir_non_overlap],
        axis=2
    )

    return fused_cube, results

# =========================================================
# RUN PIPELINE
# =========================================================

vnir_overlap_idx, swir_overlap_idx = get_overlap_bands(
    meta_data.vnir_meta["wavelength"],
    meta_data.swir_meta["wavelength"]
)

fused_cube, validation_results = fuse_vnir_swir(
    vnir_cube=vnir_cal_crop,
    swir_cube=SWIR_aligned_cube,
    vnir_waves=meta_data.vnir_meta["wavelength"],
    swir_waves=meta_data.swir_meta["wavelength"],
    vnir_overlap_idx=vnir_overlap_idx,
    swir_overlap_idx=swir_overlap_idx
)

print("Fused cube shape:", fused_cube.shape)

# =========================================================
# RGB VISUALIZATION
# =========================================================

VNIR_RGB = VNIR_synthetic_RGB(vnir_cal_crop)
SWIR_RGB = SWIR_synthetic_RGB(SWIR_aligned_cube)
FUSED_RGB = VNIR_synthetic_RGB(fused_cube)

markers = [None, None]

fig, axes = plt.subplots(1, 2, figsize=(5, 5))

images = [VNIR_RGB, SWIR_RGB]
titles = ["VNIR RGB", "SWIR RGB (False Color)"]

for i in range(2):
    axes[i].imshow(images[i])
    axes[i].set_title(titles[i])
    axes[i].axis("off")

# =========================================================
# INTERACTIVE SPECTRUM PLOT
# =========================================================

def onclick(event):
    global markers

    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    print(f"Clicked pixel: ({y}, {x})")

    for i in range(2):
        if markers[i] is not None:
            markers[i].remove()
            markers[i] = None

    for i in range(2):
        markers[i], = axes[i].plot(x, y, 'ro', markersize=6)

    fig.canvas.draw()

    vnir_spec = vnir_cal_crop[y, x, :]
    swir_spec = SWIR_aligned_cube[y, x, :]
    fused_spec = fused_cube[y, x, :]

    plt.figure(figsize=(6, 5))

    plt.plot(meta_data.vnir_meta["wavelength"], vnir_spec, label="VNIR")
    plt.plot(meta_data.swir_meta["wavelength"], swir_spec, label="SWIR")
    plt.plot(
        np.concatenate([
            meta_data.vnir_meta["wavelength"],
            meta_data.swir_meta["wavelength"]
        ]),
        fused_spec,
        label="FUSED",
        linewidth=2
    )

    plt.title(f"Spectrum at pixel ({y},{x})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid()
    plt.show()

fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
