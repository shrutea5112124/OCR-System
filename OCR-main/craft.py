# from model.craft_architecture import CRAFT
import torch
# state_dict = "C:\Users\jayv5\OneDrive\study\iiti soc\OCR\craft_mlt_25k.pth"
# device = "cuda"
# model = CRAFT().to(device)
# model.load_state_dict(state_dict)
from craft_text_detector import Craft
import os
import cv2
import numpy as np
from PIL import Image
# print(torch.cuda.is_available())  # Should print False
print(torch.version.cuda)           # Shows what CUDA version PyTorch expects
print(torch.cuda.is_available())    # Should be True if everything is fine
# print(torch.cuda.get_device_name()) #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create CRAFT object
craft = Craft(output_dir='craft_output', crop_type="box",refiner=False ,weight_path_craft_net= r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\craft_mlt_25k.pth" ,weight_path_refine_net = r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\text_detection\craft_refiner_CTW1500.pth")  # Set cuda=True if you have a GPU
path = r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\IIIT5K\train\137_6.png"
# Run text detection
prediction_result = craft.detect_text(path)
# polys = [p for p in prediction_result if p is not None and isinstance(p, (list, np.ndarray)) and len(p) > 0]


# List of bounding boxes
boxes = prediction_result["boxes"]
valid_boxes = []
for box in boxes:
    if isinstance(box, np.ndarray) and box.shape == (4, 2):
        valid_boxes.append(box)
    else:
        print("Invalid box found and skipped:", box)
print(valid_boxes)
print(valid_boxes[0].shape)
# print(boxes)


# Load the image (convert to NumPy if using PIL)
img = np.array(Image.open(path))

# Coordinates: 4 corner points (x, y) in order
# coords = np.array([
#     [x1, y1],
#     [x2, y2],
#     [x3, y3],
#     [x4, y4]
# ], dtype=np.int32)

# Create mask of same shape as image
for idx,box in enumerate(valid_boxes):
    print(box.dtype)
    # Check if box is a numpy array
    if isinstance(box, np.ndarray):
        print("very good")



    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # single channel

    #Fill polygon on mask with white (255)
    box = box.astype(np.int32)
    print(box.dtype)
    print(box.shape)
    box = np.array(box, dtype=np.int32) 
    print(box.dtype)
    cv2.fillPoly(mask, [box], 255)

    # Apply mask to the image
    masked = cv2.bitwise_and(img, img, mask=mask)

    # Crop bounding rectangle (tightest box around your polygon)
    x, y, w, h = cv2.boundingRect(box)

    # Crop that region
    cropped = masked[y:y+h, x:x+w]
    save_dir =  r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\result_images\crafttilt" 
    imag = Image.fromarray(cropped)

    # Convert to PIL for display/save
    save_path = os.path.join(save_dir, f"crop_{idx}.jpg")
    imag.save(save_path)


# Optionally unload model when done
# craft.unload_craftnet_model()
# craft.unload_refinenet_model()
