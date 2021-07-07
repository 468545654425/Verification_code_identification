import torch.nn
import torch.optim as optim
from model import get_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import cv2


batch_size = 64
def load_file(file_name):
    with open(file_name, mode='rb') as f:
        result = pickle.load(f)
    return result

char_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,\
            'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,'L':21,'M':22,\
            'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,'Z':35}

new_char_dict = {v : k for k,v in char_dict.items()}

train_data_dir = 'D:/study_data/master_1/AI/咸鱼/Verification_code_identification/data/data_code/datacode//train_data.bin'
val_data_dir = 'D:/study_data/master_1/AI/咸鱼/Verification_code_identification/data/data_code/datacode//val_data.bin'
verification_code_dir = 'D:/study_data/master_1/AI/咸鱼/Verification_code_identification/data/data_code/datacode//verification_code_data.bin'

class MyDataset(Dataset):
    def __init__(self, file_name, transforms):
        self.file_name = file_name  # 文件名称
        self.image_label_arr = load_file(self.file_name)  # 读入二进制文件
        self.transforms = transforms  # 图片转换器

    def __getitem__(self, index):
        label, img = self.image_label_arr[index]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 将图片转为灰度图

        img = cv2.medianBlur(img, ksize=5)

        img = self.transforms(img)  # 对图片进行转换
        return img, char_dict[label[0]]

    def __len__(self):
        return len(self.image_label_arr)

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([32,32]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])

train_datasets = MyDataset(train_data_dir, transform)
train_loader = DataLoader(dataset=train_datasets,batch_size=batch_size,shuffle=True)

val_datasets = MyDataset(val_data_dir, transform)
val_loader = DataLoader(dataset=val_datasets,batch_size=batch_size,shuffle=True)