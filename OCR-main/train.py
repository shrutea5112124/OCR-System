import os
import torch 
from models.dncnn import DnCNN
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from preprocess import add_gaussian_noise, add_salt_pepper_noise, add_light_blur, add_jpeg_artifacts, add_motion_blur, add_gaussian_blur_opencv, add_random_blur
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import random_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, Dataset
from models.open_cvdenoising import opencv_denoising

class DenoisingDataset(Dataset):
        def __init__(self, clean_path, transform):
            self.clean_images = [os.path.join(clean_path, f) for f in os.listdir(clean_path) if f.endswith('.png')]
            self.transform = transform
            
            

        def __len__(self):
            return len(self.clean_images)

        def __getitem__(self, idx):
            c_image = Image.open(self.clean_images[idx]).convert('L')
            noisy_img = add_salt_pepper_noise(c_image, amount=0.05, salt_vs_pepper=0.5)
            noisy_img = add_gaussian_noise(noisy_img, mean=0, sigma=0.1)
            noisy_img = add_light_blur(noisy_img, radius=1)
            noisy_img = add_jpeg_artifacts(noisy_img, quality=30) # Add Gaussian noise

            c_image = self.transform(c_image)
            n_image = self.transform(noisy_img)
            if c_image.shape[1] < 2 or c_image.shape[2] < 2:
                return None 
            return n_image, c_image

def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.default_collate(batch)
def main():
    print(f"Using device: {device}")

    model = DnCNN().to(device)

    
    # noise_path = r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\NoisyOffice\SimulatedNoisyOffice\simulated_noisy_images_grayscale"
    clean_path = r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\mixed_dataset"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 512)),
            transforms.ToTensor()
        ]
    )
    
    

    dataset = DenoisingDataset(clean_path, transform)

    # Define split sizes
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    dataloader1 = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn,num_workers=3)
    dataloader2 = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn, num_workers=6)



    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    state_dict = torch.load(r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\dncnn_finetunedfinal3.pth",map_location=device)
    model.load_state_dict(state_dict)
    model.train()

    psnr1_scores = []
    psnr2_scores = []
    ssim_scores = []
    b=os.listdir(clean_path)
    epochs = 12  # Number of epochs for training
    for epoch in range(epochs):
        for batch_idx,(n_image, c_image) in enumerate(dataloader2):
            n_image = n_image.to(device)  # Move noisy image to device
            c_image = c_image.to(device)  # Move clean image to device

            optimizer.zero_grad()  # Zero the gradients
            residual_image = model(n_image)  # Forward pass through the model
            denoised_image = n_image - residual_image  # Denoised image

            loss = criterion(denoised_image, c_image)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            if epoch % 1 ==0 and batch_idx == 0:
                output = denoised_image.cpu().detach().numpy() # Move output to CPU and detach from graph
                c_image_np = c_image.cpu().detach().numpy()  # Move clean image to CPU and detach from graph
                n_image_np = n_image.cpu().detach().numpy()  # Move noisy image to CPU and detach from graph

                for i in range(output.shape[0]):
                    psnr1 = peak_signal_noise_ratio(c_image_np[i], output[i], data_range=1.0)
                    psnr2 = peak_signal_noise_ratio(c_image_np[i], n_image_np[i], data_range=1.0)
                    ssim_score = structural_similarity(c_image_np[i][0], output[i][0], data_range=1.0)  # Grayscale channel
                    ssim_scores.append(ssim_score)
                    psnr1_scores.append(psnr1)
                    psnr2_scores.append(psnr2)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, PSNR (Denoised): {np.mean(psnr1_scores):.2f}, PSNR (Noisy): {np.mean(psnr2_scores):.2f}, SSIM: {np.mean(ssim_scores):.4f}")
                psnr1_scores.clear()
                psnr2_scores.clear() 
                ssim_scores.clear()
        torch.save( model.state_dict(), f"C:/Users/jayv5/OneDrive/study/iiti soc/OCR/preprocessing/datasets/checkpoint_epoch{epoch+1}.pth")   

    torch.save(model.state_dict(), r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\dncnn_finetunedfinal4.pth")

    state_dict = torch.load(r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\dncnn_finetunedfinal4.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    psnr_eval_scores_denoised = []
    psnr_eval_scores_noisy = []
    ssim_eval_scores = []


  
    for batch_idx,(n_test_image, c_test_image) in enumerate(dataloader1):
        
        n_test_image = n_test_image.cpu().numpy()  # Move noisy image to device
        c_test_image = c_test_image.cpu().numoy() # Move clean image to device

        with torch.no_grad():  # No need to compute gradients for evaluation

            residual_image = model(n_test_image)  # Forward pass through the model
            denoised_image = n_test_image - residual_image  # Denoised image

        output = denoised_image.cpu().numpy().squeeze(1)  # Move output to CPU and detach from graph
        c_test_image_np = c_test_image.cpu().numpy().squeeze(1)  # Move clean image to CPU and detach from graph
        n_test_image_np = n_test_image.cpu().numpy().squeeze(1)  # Move noisy image to CPU and detach from graph
        
        
        for i in range(output.shape[0]):
            psnr1 = peak_signal_noise_ratio(c_test_image_np[i], output[i], data_range=1.0)
            psnr2 = peak_signal_noise_ratio(c_test_image_np[i], n_test_image_np[i], data_range=1.0)
            ssim_score = structural_similarity(c_test_image_np[i][0], output[i][0], data_range=1.0)  # Grayscale channel
            ssim_eval_scores.append(ssim_score)
            psnr_eval_scores_denoised.append(psnr1)
            psnr_eval_scores_noisy.append(psnr2)

    print(f"Average PSNR for denoised images: {np.mean(psnr_eval_scores_denoised)}")
    print(f"Average PSNR for noisy images: {np.mean(psnr_eval_scores_noisy)}")
    print(f"Average SSIM: {np.mean(ssim_eval_scores)}")
       

    residual_image = residual_image.cpu().numpy().squeeze(1)  # Convert residual image to numpy array and remove batch dimension
    print(type(residual_image ))  # Convert residual image to numpy array and remove batch dimension
    print(type(output))      # Remove batch dimension from output
    print(type(c_test_image_np))   # Remove batch dimension from clean image
    print(type(n_test_image_np))   # Remove batch dimension from noisy image

    # if batch_idx == 0:  
    #      for i in range(output.shape[0]):   
            

    #         plt.subplot(1, 4, 1)
    #         plt.imshow(c_test_image_np[i], cmap='gray')
    #         plt.title('Clean Image')
    #         plt.axis('off')

    #         plt.subplot(1, 4, 2)
    #         plt.imshow(n_test_image_np[i], cmap='gray')
    #         plt.title('Noisy Image')
    #         plt.axis('off')

    #         plt.subplot(1, 4, 3)
    #         plt.imshow(output[i], cmap='gray')
    #         plt.title('Denoised Image')
    #         plt.axis('off')

    #         plt.subplot(1, 4, 4)
    #         plt.imshow(residual_image[i], cmap='gray')
    #         plt.title('Model Output (Residual)')
    #         plt.savefig(f'comparison_{i}.png') # Save instead of showing[4]
    #         plt.close() 

   
     
   
    
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Optional, for compatibility
    main()
    # print(f"Average SSIM: {np.mean(ssim_eval_scores)}")







# # a=os.listdir(noise_path)
# # for i in range(len(b)):
# #     if not b[i].endswith('.png'):
#         # continue  # Skip files that are not PNG images
#     # n_image_path= os.path.join(noise_path, a[i])
#     c_image_path = os.path.join(clean_path, b[i])
#     # n_image = Image.open(n_image_path).convert('L') # Convert to grayscale
#     c_image = Image.open(c_image_path).convert('L')  # Convert to grayscale
#     n_image = add_gaussian_noise(c_image, mean=0, sigma=0.1)  # Add Gaussian noise

#     n_image = transform(n_image).unsqueeze(0).to(device) # Add batch dimension
#     c_image = transform(c_image).unsqueeze(0)  # Add batch dimension
    
#     residual_image = model.forward(n_image)  # Forward pass through the
#     denoised_image = n_image - residual_image

#     optimizer.zero_grad()  # Zero the gradients
#     output = denoised_image.squeeze(0)  # Remove batch dimension
#     c_image = c_image.squeeze(0)  # Remove batch dimension
#     loss = criterion(output, c_image.to(device))  # Calculate loss
#     loss.backward()  # Backpropagation
#     optimizer.step()  # Update model parameters

#     # output = denoised_image.cpu()  # Move output to CPU for evaluation
#     # c_image = c_image.cpu()  # Move clean image to CPU for evaluation
#     # output = output.numpy().squeeze()  # Convert to numpy array and remove batch dimension
#     # c_image = c_image.numpy().squeeze()  # Convert to numpy array and remove batch dimension
#     # n_image_np= n_image.cpu().numpy().squeeze()  # Convert noisy image to numpy array and remove batch dimension
#     # residual_image = residual_image.cpu().numpy().squeeze()  # Convert residual image to numpy array and remove batch dimension
#     psnr1 = peak_signal_noise_ratio(c_image, output, data_range=1.0)
#     psnr2 = peak_signal_noise_ratio(c_image, n_image_np, data_range=1.0)
#     psnr1_scores.append(psnr1)
#     psnr2_scores.append(psnr2)
# print(f"Average PSNR for denoised images: {np.mean(psnr1_scores)}")
# print(f"Average PSNR for noisy image: {np.mean(psnr2_scores)}")

# # Assume these are your NumPy arrays:
# # c_image_np: clean image (ground truth)
# # n_image_np: noisy image
# # denoised_np: denoised image (output from your model)




# # plt.show()

# #     psnr = peak_signal_noise_ratio(c_image, output, data_range=1.0)
# #     ssim = structural_similarity(c_image, output, data_range=1.0)
# #     psnr_scores.append(psnr)
# #     ssim_scores.append(ssim)
# # print(f"Average PSNR: {np.mean(psnr_scores)}")
# # print(f"Average SSIM: {np.mean(ssim_scores)}")
# # print(output.shape)
# # print(c_image.shape)
  # print(f"Output shape: {output.shape}")
    #     #  # Convert residual image to numpy array and remove batch dimension
    # output = output.squeeze(1)  # Remove batch dimension
    # c_test_image_np = c_test_image_np.squeeze(1)  # Remove batch dimension  
    # n_test_image_np = n_test_image_np.squeeze(1) 
    # residual_image = residual_image.cpu().detach().numpy().squeeze(1)

   