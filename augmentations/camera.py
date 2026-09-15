#!/opt/homebrew/opt/python@3.14/bin/python3.14
# camera augmentations 

import math
import cv2
import sys
import os 
import numpy as np

def motion_kernel(length=11, angle_deg=0):
    n = int(max(3, round(length)))
    k = np.zeros((n, n), np.float32)
    c = (n - 1) / 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    p0 = (round(c - dx*(n-1)/2), round(c - dy*(n-1)/2))
    p1 = (round(c + dx*(n-1)/2), round(c + dy*(n-1)/2))
    cv2.line(k, p0, p1, 1.0, 1)
    if k.sum() == 0:
        k[n//2, n//2] = 1
    return k / k.sum()

def motion_blur(img, length=3, angle=10):
    k = motion_kernel(length, angle)
    return np.stack([cv2.filter2D(img[...,c], -1, k)
                     for c in range(3)], axis=-1)

def disk_kernel(radius):
    r = int(max(1, round(radius)))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    k = ((xx*xx + yy*yy) <= r*r).astype(np.float32)
    return k / k.sum()

def defocus(img, radius=1):
    k = disk_kernel(radius)
    return np.stack([cv2.filter2D(img[...,c], -1, k)
                     for c in range(3)], axis=-1)

def poisson_read_noise(img, photons=1200, sigma_read=0.013):
    # Poisson shot noise + Gaussian read noise.
    y = np.clip(img, 0, 1) * photons
    y = np.random.poisson(y).astype(np.float32) / photons
    y += np.random.normal(0, sigma_read, img.shape).astype(np.float32)
    return np.clip(y, 0, 1)

def radial_distortion(img, k1=-0.03, k2=0.015):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx-(w-1)/2)/((w-1)/2)
    yn = (yy-(h-1)/2)/((h-1)/2)
    r2 = xn*xn + yn*yn
    s = 1 + k1*r2 + k2*r2*r2
    mx = ((xn*s)*(w-1)/2 + (w-1)/2).astype(np.float32)
    my = ((yn*s)*(h-1)/2 + (h-1)/2).astype(np.float32)
    return np.stack([
        cv2.remap(img[...,c], mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT101)
        for c in range(3)
    ], axis=-1)

def vignette(img, alpha=0.008):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h,0:w].astype(np.float32)
    xx = (xx-(w-1)/2)/((w-1)/2)
    yy = (yy-(h-1)/2)/((h-1)/2)
    v = np.exp(-alpha*(xx*xx+yy*yy))[...,None]
    return np.clip(img*v, 0, 1)

def jpeg(img, quality=50):
    bgr = cv2.cvtColor((np.clip(img,0,1)*255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)/255

def real_estate_degradation(img):
    """
    img: float32 RGB in [0,1].
    Simulates a plausible interior capture:
    wide-angle distortion + CA + slight camera shake + sensor noise
    + vignetting + exposure loss + JPEG.
    """
    x = img.astype(np.float32) / 255 if img.dtype != np.float32 else img
    x = radial_distortion(x, k1=np.random.uniform(-0.16,-0.08))
    x = motion_blur(x,
                    length=np.random.uniform(3,9),
                    angle=np.random.uniform(-25,25))
    x = poisson_read_noise(
        x,
        photons=np.random.uniform(80,280),
        sigma_read=np.random.uniform(0.006,0.020)
    )
    x = vignette(x, alpha=np.random.uniform(0.15,0.55))
    x = np.clip(x*np.random.uniform(0.78,1.0), 0, 1)
    return jpeg(x, quality=np.random.randint(40,81))


def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

def save_rgb(path, img):
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr):
        raise RuntimeError(f"write failed: {path}")

def random_flip(img):
    return np.flip(img, axis=1)

AUGMENTATIONS = {
    # "motion_blur": lambda x: motion_blur(x, length=7, angle=10),
    # "defocus": lambda x: defocus(x, radius=4),
    "poisson_read_noise": lambda x: poisson_read_noise(x),
    # "radial_distortion": lambda x: radial_distortion(x),
    # "vignette": lambda x: vignette(x, alpha=0.3),
    # "jpeg": lambda x: jpeg(x, quality=50),
    # "real_estate_degradation": real_estate_degradation,
    "random_flip" : lambda x: random_flip(x)
}

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.abspath(__file__))
    rel_image_path = '../images/image1.jpg'
    image_path = os.path.join(cwd, rel_image_path)
    out_dir = os.path.join('/tmp', 'augmentations')
    os.makedirs(out_dir, exist_ok=True)

    img = load_rgb(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    for name, fn in AUGMENTATIONS.items():
        out = fn(img)
        out_path = os.path.join(out_dir, f"{stem}_{name}.jpg")
        save_rgb(out_path, out)
        print(f"{name} -> {out_path}")