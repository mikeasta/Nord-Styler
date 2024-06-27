import cv2 as cv
import numpy as np
import pathlib
from PIL import ImageColor

IMAGE_DIR             = pathlib.Path("../../../images/")
IMAGE_LABEL           = "kiryu_nishiki.jpg"

USE_THRESHOLDING    = True
THRESHOLDING_KERNEL = 3
MORPHOLOGY_KERNEL   = 3

__IMAGE_READ_PATH       = IMAGE_DIR / pathlib.Path(IMAGE_LABEL)
__IMAGE_SAVE_PATH       = IMAGE_DIR / pathlib.Path(f"nord_{IMAGE_LABEL}")
__IMAGE_SAVE_MORPH_PATH = IMAGE_DIR / pathlib.Path(f"nord_morph_transforms_{IMAGE_LABEL}")


# Nord theme colors
# ! PICK THOSE WHICH U WANT TO SEE IN THE RESULT IMAGE
# You can preview colors in VSCode, using extension below:
# https://marketplace.visualstudio.com/items?itemName=wilsonsio.color-vision
hex_colors = [
    "#2e3440", # --- 
    "#3b4252", #   | [Polar Night] 
    "#434c5e", #   | Shades of dark gray
    "#4c566a", # ---
    "#d8dee9", # ---
    "#e5e9f0", #   | [Snow Storm] - Shades of white
    "#eceff4", # ---
    "#5e81ac", # ---                        
    "#81a1c1", #   | [Frost]                
    "#88c0d0", #   | Shades of ice colors   
    "#8fbcbb", # ---                        
    "#b48ead", # ---
    "#d08770", #   | [Aurora]
    "#bf616a", #   | Shades of warm colors
    "#ebcb8b", #   |
    "#a3be8c"  # ---
]

# Convert
rgb_colors = [np.asarray(ImageColor.getcolor(hex_color, "RGB")) for hex_color in hex_colors]

# Load image
img = cv.imread(__IMAGE_READ_PATH)

# Apply adaptive thresholding + gaussian blue
def apply_threshold(image: np.array, kernel_size: int = 3):
    # Split to several channels
    b, g, r = cv.split(image)
    kernel = (kernel_size, kernel_size)

    b_blur = cv.GaussianBlur(b,kernel,0)
    g_blur = cv.GaussianBlur(g,kernel,0)
    r_blur = cv.GaussianBlur(r,kernel,0)
    
    b_ret, b_th = cv.threshold(b_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    g_ret, g_th = cv.threshold(g_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    r_ret, r_th = cv.threshold(r_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    return cv.merge((b_th, g_th, r_th))


# Distance between two 3D vectors
def distance(vec1, vec2):
    return np.sqrt(
        (vec1[0] - vec2[0]) ** 2 + 
        (vec1[1] - vec2[1]) ** 2 +
        (vec1[2] - vec2[2]) ** 2
    )


# Get color for current pixel
def get_color(current_color):
    current_color = current_color[::-1]
    distances = [distance(current_color, style_color) for style_color in rgb_colors]
    nearest_color = rgb_colors[np.argmin(distances)]
    return nearest_color[::-1]


# Apply transform
def colorize(img):
    rows, cols, ch = img.shape
    new_img = np.zeros((rows, cols, ch))
    for r in range(rows):
        for c in range(cols):
            new_img[r][c] = get_color(img[r][c])
    return new_img


# Open morphological transform
def open_morph(image: np.array, kernel_size: int = MORPHOLOGY_KERNEL):
    # Split to several channels
    b, g, r = cv.split(image)

    # Errosion to each channel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    b_eroded = cv.erode(b, kernel, iterations=1)
    g_eroded = cv.erode(g, kernel, iterations=1)
    r_eroded = cv.erode(r, kernel, iterations=1)

    # Dilation to each channel
    b_dilated = cv.dilate(b_eroded, kernel, iterations=1)
    g_dilated = cv.dilate(g_eroded, kernel, iterations=1)
    r_dilated = cv.dilate(r_eroded, kernel, iterations=1)

    # Merge channels to get result image
    return cv.merge((b_dilated, g_dilated, r_dilated))


# Close morphological transform
def close_morph(image: np.array, kernel_size: int = MORPHOLOGY_KERNEL):
    # Split to several channels
    b, g, r = cv.split(image)

    # Errosion to each channel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    b_dilated = cv.dilate(b, kernel, iterations=1)
    g_dilated = cv.dilate(g, kernel, iterations=1)
    r_dilated = cv.dilate(r, kernel, iterations=1)

    # Dilation to each channel
    b_eroded = cv.erode(b_dilated, kernel, iterations=1)
    g_eroded = cv.erode(g_dilated, kernel, iterations=1)
    r_eroded = cv.erode(r_dilated, kernel, iterations=1)

    # Merge channels to get result image
    return cv.merge((b_eroded, g_eroded, r_eroded))


# Apply threshold transform
if USE_THRESHOLDING:
    img = apply_threshold(img, kernel_size=THRESHOLDING_KERNEL)

styled = colorize(img)

# Apply morph transform
styled_open_morph = open_morph(styled, kernel_size=MORPHOLOGY_KERNEL)
styled_both_morph = close_morph(styled_open_morph, kernel_size=MORPHOLOGY_KERNEL)

# Create demo image
demo_img = styled / 255
cv.imshow("PRESS ANY BUTTON TO SAVE", demo_img)
cv.waitKey(0)

# Save images
cv.imwrite(__IMAGE_SAVE_PATH, styled)
cv.imwrite(__IMAGE_SAVE_MORPH_PATH, styled_both_morph)

cv.destroyAllWindows()
