import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
# 数据文件夹路径
data_folder = '../'
data_folder2 = '../data'
# 读取训练数据列表
train_file = os.path.join(data_folder, 'train.txt')
with open(train_file, 'r') as f:
    train_data = f.readlines()[1:]  # 跳过第一行

# 提取数字的取值范围和情感标签
num_range = []
sentiments = []
for line in train_data:
    num, sentiment = line.strip().split(',')
    num_range.append(int(num))
    sentiments.append(sentiment)

# 转换情感标签为0、1、2
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
sentiment_labels = [sentiment_mapping[sentiment] for sentiment in sentiments]

# 图像数据
image_data = []
text_data = []
widths = []
heights = []
# 图像EDA
for num, sentiment in tqdm(zip(num_range, sentiment_labels), total=len(num_range), desc='Processing Images'):
    # 处理图像
    image_name = os.path.join(data_folder2, f'{num}.jpg')
    image = Image.open(image_name)
    width, height = image.size
    widths.append(width)
    heights.append(height)
    text_name = os.path.join(data_folder2, f'{num}.txt')
    with open(text_name, 'r',encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()
    text_data.append(text)

    # 在此处进行图像的EDA操作，例如绘制直方图、散点图等
    # 这里使用Seaborn库绘制图像的直方图


    # 添加图像数据
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    image = image_transform(image)
    image_data.append(image)

# 转换为PyTorch的Tensor
image_tensor = torch.stack(image_data)


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
encoding = tokenizer.batch_encode_plus(text_data, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
input_ids = encoding['input_ids']

file_path = "../data"
result_tensor = torch.tensor(sentiment_labels)
torch.save(input_ids, file_path+"/text.pt")
torch.save(image_tensor, file_path+"/image.pt")
torch.save(result_tensor, file_path+"/result.pt")
# EDA
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(widths)
plt.title('Image Width Distribution')
plt.xlabel('Width')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(heights)
plt.title('Image Height Distribution')
plt.xlabel('Height')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
text_lengths = [len(text) for text in text_data]
plt.figure()
sns.histplot(text_lengths)
plt.title('Text Length Distribution')
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()
train_file = os.path.join(data_folder, 'test_without_label.txt')
with open(train_file, 'r') as f:
    train_data = f.readlines()[1:]  # 跳过第一行

# 提取数字的取值范围和情感标签
num_range = []
sentiments = []
for line in train_data:
    num, sentiment = line.strip().split(',')
    num_range.append(int(num))
    sentiments.append(sentiment)
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2,'null':3}
sentiment_labels = [sentiment_mapping[sentiment] for sentiment in sentiments]

# 图像数据
image_data = []
text_data = []
for num, sentiment in tqdm(zip(num_range, sentiment_labels), total=len(num_range), desc='Processing Images'):
    # 处理图像
    image_name = os.path.join(data_folder2, f'{num}.jpg')
    image = Image.open(image_name)
    text_name = os.path.join(data_folder2, f'{num}.txt')
    with open(text_name, 'r',encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()
    text_data.append(text)

    # 在此处进行图像的EDA操作，例如绘制直方图、散点图等
    # 这里使用Seaborn库绘制图像的直方图


    # 添加图像数据
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    image = image_transform(image)
    image_data.append(image)

# 转换为PyTorch的Tensor
image_tensor = torch.stack(image_data)
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
encoding = tokenizer.batch_encode_plus(text_data, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
input_ids = encoding['input_ids']
file_path = "../data"
torch.save(input_ids, file_path+"/text_test.pt")
torch.save(image_tensor, file_path+"/image_test.pt")