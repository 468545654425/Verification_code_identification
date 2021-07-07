import torch
from dataset import load_file,verification_code_dir,transform,new_char_dict
import torch.nn as nn
import torch.optim as optim
from model import get_model
from torchvision import datasets, transforms
from dataset import train_loader,val_loader
import matplotlib.pyplot as plt
import PIL as Image
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    loss_records_list = []
    # 这里作为一个选择，因为以后可能有更多的backbone，那么我们只需要加入到此列表即可

    verification_code_data = load_file(verification_code_dir)
    image = verification_code_data[6]
    img = cv2.medianBlur(image.copy(), 5)
    plt.imshow(img)
    plt.show()
    for backbone_type in ['lenet', 'resnet', 'vgg']:
        print("使用{}网络".format(backbone_type))
        check_net = get_model(backbone_type)
        check_net.load_state_dict(torch.load('{}.tar'.format(backbone_type)))
        IMAGES = list()
        NUMS = list()
        for img in verification_code_data:

            IMAGES.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            image_1 = img[:, :80]
            image_2 = img[:, 80:160]
            image_3 = img[:, 160:240]
            image_4 = img[:, 240:320]
            img_list = [image_1, image_2, image_3, image_4]

            nums = []
            for one_img in img_list:

                one_img = transform(one_img)
                one_img = one_img.unsqueeze(0)
                output = check_net(one_img)
                nums.append(new_char_dict[torch.argmax(output).item()])
            NUMS.append('Verification_code({}) : '.format(backbone_type) + ''.join(nums))

        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        for i in range(1, 11):
            plt.subplot(5, 2, i)
            plt.title(NUMS[i - 1], fontsize=25, color='red')
            plt.imshow(IMAGES[i - 1])
            plt.xticks([])
            plt.yticks([])
        plt.show()