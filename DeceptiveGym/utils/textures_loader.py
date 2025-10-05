import os
from PIL import Image
import numpy

import numpy as np

def enhance_saturation(img, factor=1.2):
    """
    Enhance saturation of an image (RGB, CHW format, float32 in [0,1]).

    Args:
        img (np.ndarray): Image array of shape (3, H, W), values in [0,1].
        factor (float): Factor to scale saturation (>1 increases, <1 decreases).

    Returns:
        np.ndarray: Saturation-enhanced image (3, H, W), float32 in [0,1].
    """
    assert img.ndim == 3 and img.shape[0] == 3, "Input must be (3, H, W)"
    
    # Transpose to HWC
    img_hwc = np.transpose(img, (1, 2, 0))

    # --- RGB -> HSV ---
    cmax = img_hwc.max(axis=-1)
    cmin = img_hwc.min(axis=-1)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(cmax)
    mask = delta > 1e-6
    r, g, b = img_hwc[..., 0], img_hwc[..., 1], img_hwc[..., 2]

    idx = (cmax == r) & mask
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (cmax == g) & mask
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = (cmax == b) & mask
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    hue = hue / 6.0  # normalize [0,1]

    # Saturation
    sat = np.zeros_like(cmax)
    sat[mask] = delta[mask] / cmax[mask]

    # Value
    val = cmax

    # Enhance saturation
    sat = np.clip(sat * factor, 0, 1)

    # --- HSV -> RGB ---
    h = hue * 6
    c = val * sat
    x = c * (1 - np.abs((h % 2) - 1))
    m = val - c

    rgb = np.zeros_like(img_hwc)
    h_idx = (h.astype(int) % 6)

    # Instead of tuples with scalars, do per-case assignments
    for k in range(6):
        mask_k = h_idx == k
        if k == 0:
            rgb[mask_k] = np.stack([c[mask_k], x[mask_k], np.zeros_like(c[mask_k])], axis=-1)
        elif k == 1:
            rgb[mask_k] = np.stack([x[mask_k], c[mask_k], np.zeros_like(c[mask_k])], axis=-1)
        elif k == 2:
            rgb[mask_k] = np.stack([np.zeros_like(c[mask_k]), c[mask_k], x[mask_k]], axis=-1)
        elif k == 3:
            rgb[mask_k] = np.stack([np.zeros_like(c[mask_k]), x[mask_k], c[mask_k]], axis=-1)
        elif k == 4:
            rgb[mask_k] = np.stack([x[mask_k], np.zeros_like(c[mask_k]), c[mask_k]], axis=-1)
        elif k == 5:
            rgb[mask_k] = np.stack([c[mask_k], np.zeros_like(c[mask_k]), x[mask_k]], axis=-1)

    rgb += m[..., None]

    # Back to CHW
    rgb = np.transpose(rgb, (2, 0, 1))
    return np.clip(rgb, 0, 1).astype(np.float32)


def load_textures(root_dir, height, width):
    textures = []
    texture_dict = {}
    index = 0

    # Go through each subfolder = texture category
    for category in sorted(os.listdir(root_dir)):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue  # skip files at root level
        
        texture_dict[category] = []
        
        # Process each image in the category
        for fname in sorted(os.listdir(category_path)):
            fpath = os.path.join(category_path, fname)
            try:
                with Image.open(fpath) as img:
                    print("loadinf ", fpath)
                    # Ensure RGB
                    img = img.convert("RGB")
                    img = img.resize((width, height), Image.BILINEAR)
                    arr = numpy.asarray(img, dtype=numpy.float32) / 255.0  # HWC, range 0..1
                    arr = numpy.transpose(arr, (2, 0, 1))  # CHW
                    
                    
                    
                    textures.append(arr)
                    
                    texture_dict[category].append(index)
                    index += 1
            except Exception as e:
                print(f"Skipping {fpath}, error: {e}")


        

    if textures:
        textures = numpy.stack(textures, axis=0)  # (N,3,H,W)
    else:
        textures = numpy.empty((0, 3, height, width), dtype=numpy.float32)


    #for n in range(len(textures)):
    #    textures[n] = enhance_saturation(textures[n], 2.0)

    return textures, texture_dict