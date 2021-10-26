#! /usr/bin/env python

import pickle
import albumentations as A
import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
import face_recognition
import os
import matplotlib.pyplot as plt
import pandas

test_chip = '/mnt/NAS/Photos/pkls/assigned/batch1/imchip_326465.pkl'
# test_chip = '/mnt/NAS/Photos/pkls/assigned/batch26/imchip_591528.pkl'


def short_vec(image):
    h, w = image.shape[:2]
    bbox = [(0, w, h, 0)]
    encoding = face_recognition.face_encodings(image, known_face_locations = bbox, num_jitters=400, model='large')[0]
    return encoding

def long_vec(image):

    assert np.abs(image.shape[0] - image.shape[1]) < 3, f"{image.shape[0]} != {image.shape[1]}"
    image = cv2.resize(image, (160, 160))
    image = np.moveaxis(image, 2, 0)

    image = (image - 127.5) / 128
    image = torch.Tensor(image)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    encoding = resnet(image.unsqueeze(0))
    encoding = encoding.detach().numpy().astype(np.float32)
    encoding = list(encoding[0])

    return np.array(encoding)

transform = A.Compose([
    A.geometric.transforms.Perspective(scale=(0.05, 0.1)),
    A.geometric.rotate.Rotate(limit=25),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.05),
    A.ColorJitter(hue=0.03, contrast=0.2),
    A.RandomFog(p=0.05), ### 
    A.Blur(blur_limit=11),
    A.GaussNoise(var_limit=(10.0, 80.0), p=0.5),
    A.CLAHE( p=0.32, clip_limit=(2,3)),

    # A.geometric.transforms.ElasticTransform(),
    # A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0.4), contrast_limit=(-0.2, 0.2),), ### 
    # A.Flip(),
    # # A.ToGray(),
    # A.Downscale( p=0.4),
    # A.FancyPCA( p=1),
])


def random_center_crop(image, w, h):

    oversize = np.random.rand() / 4 + 1
    im_h, im_w = image.shape[:2]
    c = np.array(image.shape[:2]) // 2
    adj_shape = np.min(image.shape[:2]) * 0.03
    c_adj = (np.random.randn(2) * adj_shape ).astype(np.int)
    c = c + c_adj
    l = int(c[0] - w * oversize)
    r = int(c[0] + w * oversize)
    t = int(c[1] - h * oversize)
    b = int(c[1] + h * oversize)

    t = np.max( (0, t) )
    l = np.max( (0, l) )
    b = np.min( (b, im_h) )
    r = np.min( (r, im_w) )

    new_w = np.abs(l - r)
    new_h = np.abs(b - t)

    im = image[t:b, l:r, :]

    if np.abs(new_w - new_h) > 1:

        # Recenter in new chip
        c = np.array(im.shape[:2]) // 2
        size = np.min((new_w, new_h)) // 2

        t_prime = int(c[0] - size)
        b_prime = int(c[0] + size)
        l_prime = int(c[1] - size)
        r_prime = int(c[1] + size)


        im = im[t_prime:b_prime, l_prime:r_prime, :]
    # print(im.shape)

    return im


def augment_image(data_dict, n_iters = 5):
    image = data_dict['chipped_image']
    w = data_dict['width'] // 2
    h = data_dict['height'] // 2

    max_extent = max(data_dict['width'], data_dict['height'])
    scale_up_size = int(np.ceil(np.sqrt(2) * max_extent))
    scale_down = 800 / scale_up_size

    # chip_h = data_dict['height'] * np.sqrt(2)
    # chip_w = data_dict['width'] * np.sqrt(2)
    if scale_up_size > 800: 
        # Scale down and re-calculate bounding box. 
        center = np.array(image.shape[:2]) // 2

        width = w * scale_down
        height = h * scale_down

        top = int(center[0] - height)
        bot = int(center[0] + height)
        lef = int(center[1] - width)
        rig = int(center[1] + width)

        w = np.ceil(width)
        h = np.ceil(height)


    w = h = int(np.min((w, h)))

    shorts = []
    longs = []
    for ii in range(n_iters):
        im = random_center_crop(image, w, h)
        transformed = transform(image=im) # , cropping_bbox=[l, t, w * 2, h * 2])
        transformed_image1 = transformed["image"]

        enc_512 = long_vec(transformed_image1).astype(np.float16)
        enc_128 = short_vec(transformed_image1).astype(np.float16)
        if len(enc_512) == 512 and len(enc_128) == 128:
            longs.append(enc_512)
            shorts.append(enc_128)

    return shorts, longs


with open(test_chip, 'rb') as fh :
    print(test_chip)
    data = pickle.load(fh)

    image = data['chipped_image']

    w = data['width'] // 2
    h = data['height'] // 2

    max_extent = max(data['width'], data['height'])
    scale_up_size = int(np.ceil(np.sqrt(2) * max_extent))
    scale_down = 800 / scale_up_size

    # chip_h = data_dict['height'] * np.sqrt(2)
    # chip_w = data_dict['width'] * np.sqrt(2)
    if scale_up_size > 800: 
        # Scale down and re-calculate bounding box. 
        center = np.array(image.shape[:2]) // 2

        width = w * scale_down
        height = h * scale_down

        top = int(center[0] - height)
        bot = int(center[0] + height)
        lef = int(center[1] - width)
        rig = int(center[1] + width)

        w = np.ceil(width)
        h = np.ceil(height)


    w = h = int(np.min((w, h)))

    # v = long_vec(image)
    # vs = short_vec(image)
    # v2 = long_vec(np.fliplr(image))

    fig, axs = plt.subplots(5, 5, figsize=(20, 15))

    plt.style.use('ggplot')
    axs[0, 0].imshow(image)
    axs[0, 0].axis('off')

    for i in range(24):
        ax_x = (i + 1) % 5
        ax_y = (i + 1) // 5
        im = random_center_crop(image, w, h)
        transformed = transform(image=im) # , cropping_bbox=[l, t, w * 2, h * 2])
        transformed_image1 = transformed["image"]
        # print(long_vec(transformed_image1))
        axs[ax_y, ax_x].imshow(transformed_image1)
        axs[ax_y, ax_x].axis('off')
        
    fig.tight_layout()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()

base = '/mnt/NAS/Photos/pkls/assigned/batch1/'
for root, dirs, files in os.walk(base):
    for f in files[:10]:
        fullfile = os.path.join(root, f)
        print(fullfile)

        with open(fullfile, 'rb') as fh:
            data = pickle.load(fh)


        exit()

        shorts, longs = augment_image(data, 2)