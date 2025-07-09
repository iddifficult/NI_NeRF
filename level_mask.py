import  torch
import numpy as np


def get_mask(mlp, coordinate, mask, encoder_dim=32):

    w,h,l = coordinate.shape[:3]

    high_pass = np.ones((w, h), np.uint8)
    for i in range(w):
        for j in range(h):
            if (i - w//2) ** 2 + (j - h//2) ** 2 < (w//4) ** 2:
                high_pass[i, j] = 0

    with torch.no_grad():
        for i in range(encoder_dim):
            slice_l = coordinate[:,:,l//2,:].reshape(-1,3)
            img_jpg_l = mlp.encoder(slice_l)[:,i]
            img_jpg_l = img_jpg_l.reshape(w,h).float().cpu().detach().numpy()
            img_jpg = img_jpg_l
            max = np.max(img_jpg)
            min = np.min(img_jpg)
            img_jpg = (img_jpg-min)/(max-min)

            f = np.fft.fft2(img_jpg)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.clip(np.abs(fshift)*high_pass,0,1)
            energy = np.mean(magnitude_spectrum ** 2)
            if energy>0.8:
                mask[:,i:] = 0
                break

    return mask
            