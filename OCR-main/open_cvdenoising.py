import cv2
import numpy as np

def opencv_denoising(img_path):


    # Step 1: Salt & pepper noise removal (apply always)
    img = cv2.medianBlur(img_path, 3)

    # Step 2: Gaussian noise removal (apply always)
    new_img = cv2.fastNlMeansDenoising (img , None, h=10, templateWindowSize=7, searchWindowSize=21)

    

    # Step 3: Blur detection
    def is_blurry(img, threshold=100):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

    # Step 4: Apply sharpening only if blurry
    is_grayscale = len(img.shape) == 2 or img.shape[2] == 1
    
    # Convert grayscale to BGR if needed
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

    # Step 5: Resize the image to 640x640
    def smart_resize(img, max_size=640, min_size=240):
        h, w = img.shape[:2]

        # If image is already large enough, keep it unchanged
        if max(h, w) >= max_size:
            return img

        # Compute scale to resize while keeping aspect ratio
        scale = min(max_size / max(h, w), min_size / min(h, w))
        
        # Only upscale slightly if it's very small
        if scale < 1.0:
            return img  # Don't downscale
        elif scale > 2.0:
            scale = 4.0  # Limit over-enlargement

        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # step 6 : Convert to grayscale if not already

    new_img = smart_resize(new_img)

    if is_blurry(new_img):
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        new_img = cv2.filter2D(new_img, -1, kernel)

    if len(new_img.shape) == 3 and new_img.shape[2] == 3:
            new_img= cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    # Convert image from grayscale to binary
    

        

    return new_img
