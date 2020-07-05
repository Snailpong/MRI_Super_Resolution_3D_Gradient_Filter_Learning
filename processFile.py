import os
import numpy as np
import nibabel as nib

crop_pass = 40

def def processile():
    HR = mat

    mat_max = np.max(mat)
    mat = mat / mat_max

    # Fuzzy LR
    # Upscaling
    # cv2.GaussianBlur()
    # LR = cv2.GaussianBlur(LR,(0,0),2);

    fig = plt.figure()

    img = np.transpose(mat, (1, 0, 2))

    fig.add_subplot(2, 3, 1)
    plt.imshow(img[:, :, 130])

    img = np.fft.fftshift(img)

    imgfft = np.fft.fftn(img)
    imgfft = np.fft.fftshift(imgfft)

    fig.add_subplot(2, 3, 2)
    plt.imshow(20 * np.log(abs(imgfft[:, :, 0])))  # magnitude

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_cut = np.array(imgfft)
    imgfft_cut[x_center - crop_pass: x_center + crop_pass,
    y_center - crop_pass: y_center + crop_pass, z_center - crop_pass: z_center + crop_pass] = 0
    imgfft = imgfft - imgfft_cut

    fig.add_subplot(2, 3, 3)  # substracted magnitude
    plt.imshow(20 * np.log(abs(imgfft[:, :, 130])))
    imgifft = np.fft.ifftn(imgfft)

    fig.add_subplot(2, 3, 4)  # ifft
    plt.imshow(abs(imgifft[:, :, 0]))

    fig.add_subplot(2, 3, 5)  # 원래와 비교한 것
    plt.imshow(img[:, :, 0] - np.abs(imgifft[:, :, 0]))

    fig.add_subplot(2, 3, 6)
    imgifft = np.concatenate((imgifft[x_center:, :, :], imgifft[:x_center, :, :]), axis=0)
    imgifft = np.concatenate((imgifft[:, y_center:, :], imgifft[:, :y_center, :]), axis=1)
    imgifft = np.concatenate((imgifft[:, :, z_center:], imgifft[:, :, :z_center]), axis=2)

    LR = np.abs(imgifft)
    LR = LR / np.max(LR) * mat_max

    plt.imshow(LR[:, :, 130])

    LR = np.transpose(LR, (1, 0, 2))
    LR = LR.astype('float32')

    print(HR[x_center - crop_pass: x_center + crop_pass, y_center - crop_pass: y_center + crop_pass, 130])
    print(LR[x_center - crop_pass: x_center + crop_pass, y_center - crop_pass: y_center + crop_pass, 130])

    print(mat_max)

    print(np.min(LR))
    print(np.max(LR))

    plt.show()

    ni_img = nib.Nifti1Image(LR, np.eye(4))
    nib.save(ni_img, 'output.nii.gz')