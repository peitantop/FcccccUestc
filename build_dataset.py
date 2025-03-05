from torch.utils.data import DataLoader
import torch
from torchvision import transforms, datasets
import pandas as pd

data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'


def csv_to_vectors_pandas(csv_file, columns=None):
    # 有标题行，设置 header=0
    df = pd.read_csv(csv_file, header=0, encoding='windows-1254')
    if columns is not None:
        df = df.iloc[:, columns]
    vectors = df.values.astype(int).tolist()
    return vectors

# # 示例：读取所有列
# vectors = csv_to_vectors_pandas('data.csv')
# # 示例：读取前两列
vectors = csv_to_vectors_pandas(label_dir, columns=[7, 14])
print(vectors)

batch_size = 32

# 图像变换
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=1.1, contrast=1.5, saturation=0.8),       # 可根据训练结果调节
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=35),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# -------------------------------------------------------------------------------------------------

# 数据集划分（80%训练，20%验证）

dataset = datasets.ImageFolder(root=data_dir, transform=None)
data_len = len(dataset)

train_size = int(0.8 * data_len)
val_size = data_len - train_size
train_indices, val_indices = torch.utils.data.random_split(
    dataset=dataset, lengths=[train_size, val_size]
)


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


train_dataset = TransformSubset(train_indices, transform=train_transforms)
val_dataset = TransformSubset(val_indices, transform=valid_transforms)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
