import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertForSequenceClassification
import torch.optim as optim
from tqdm import tqdm
file_path = "../data"
train_input_ids=torch.load(file_path+"/text.pt")
train_image_tensor= torch.load(file_path+"/image.pt")
train_result_tensor= torch.load(file_path+"/result.pt")
val_input_ids=torch.load(file_path+"/text_test.pt")
val_image_tensor= torch.load(file_path+"/image_test.pt")
class classify(nn.Module):
    def __init__(self):
        super(classify, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2000, 500)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(500, 3)
        self.FC = nn.Linear(768,1000)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,y):
        x = self.resnet(x)
        y = self.bert(y)
        y = y.last_hidden_state[:, 0, :]
        y = self.FC(y)
        z=torch.cat((x, y), dim=1)
        z=self.fc1(z)
        z=self.relu(z)
        z = self.dropout(z) if self.training else z
        z=self.fc2(z)

        return z
class classify2(nn.Module):
    def __init__(self):
        super(classify2, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1000, 100)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(100, 3)
        self.FC = nn.Linear(768,1000)
        self.weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,y):
        x = self.resnet(x)
        y = self.bert(y)
        y = y.last_hidden_state[:, 0, :]
        y = self.FC(y)
        weighted_sum = self.weight[0] * x + self.weight[1] * y
        z=self.fc1(weighted_sum)
        z=self.relu(z)
        z = self.dropout(z) if self.training else z
        z=self.fc2(z)

        return z
class classify3(nn.Module):
    def __init__(self):
        super(classify3, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=6, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1000, 300)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(300, 3)
        self.FC = nn.Linear(768,300)
        self.fc3 = nn.Linear(6,3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,y):
        x = self.resnet(x)
        y = self.bert(y)
        y = y.last_hidden_state[:, 0, :]
        x=self.fc1(x)
        x=self.relu(x)
        x = self.dropout(x) if self.training else x
        x=self.fc2(x)
        y=self.FC(y)
        y=self.relu(y)
        y = self.dropout(y) if self.training else y
        y=self.fc2(y)
        z=torch.cat((x, y), dim=1)
        z = self.transformer_encoder(z)
        #z = torch.mean(z, dim=1)
        z=self.fc3(z)


        return z
class classify4(nn.Module):
    def __init__(self):
        super(classify4, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1000, 500)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(500, 3)
        self.FC = nn.Linear(768,1000)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,y):
        z = self.resnet(x)
        z=self.fc1(z)
        z=self.relu(z)
        z = self.dropout(z) if self.training else z
        z=self.fc2(z)

        return z
class classify5(nn.Module):
    def __init__(self):
        super(classify5, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1000, 500)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(300, 3)
        self.FC = nn.Linear(768,300)
        self.dropout = nn.Dropout(0.5)

    def forward(self, z):
        z = self.bert(z)
        z = z.last_hidden_state[:, 0, :]
        z=self.FC(z)
        z=self.relu(z)
        z = self.dropout(z) if self.training else z
        z=self.fc2(z)

        return z
torch.cuda.empty_cache()
model1 = classify().train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model1.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model1(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = classify().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model2.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model2(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model3 = classify().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model3.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
criterion = nn.MultiMarginLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model3(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model4 = classify().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model4.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.MultiMarginLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model4(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
torch.cuda.empty_cache()
model5 = classify2().train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model5.to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model5.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model5(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model6 = classify2().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model6.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model6(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model7 = classify2().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model7.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
criterion = nn.MultiMarginLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model7(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model8 = classify2().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model8.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.MultiMarginLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model8(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
torch.cuda.empty_cache()
model9 = classify3().train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model9.to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model9.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model9(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model10 = classify3().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model10.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model10(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model11 = classify3().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model11.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
criterion = nn.MultiMarginLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model11(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model12 = classify3().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model12.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.MultiMarginLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model12(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model13 = classify4().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model13.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model13(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model14 = classify5().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model14.parameters(), lr=0.0001)
num_epochs = 20
batch_size = 100
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 2.8, 0.5], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model14(batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
# 保存 model1 的参数
torch.save(model1.state_dict(), '../data/model1.pth')

# 保存 model2 的参数
torch.save(model2.state_dict(), '../data/model2.pth')

# 保存 model3 的参数
torch.save(model3.state_dict(), '../data/model3.pth')

# 保存 model4 的参数
torch.save(model4.state_dict(), '../data/model4.pth')

# 保存 model5 的参数
torch.save(model5.state_dict(), '../data/model5.pth')

# 保存 model6 的参数
torch.save(model6.state_dict(), '../data/model6.pth')

# 保存 model7 的参数
torch.save(model7.state_dict(), '../data/model7.pth')

# 保存 model8 的参数
torch.save(model8.state_dict(), '../data/model8.pth')

# 保存 model9 的参数
torch.save(model9.state_dict(), '../data/model9.pth')

# 保存 model10 的参数
torch.save(model10.state_dict(), '../data/model10.pth')

# 保存 model11 的参数
torch.save(model11.state_dict(), '../data/model11.pth')

# 保存 model12 的参数
torch.save(model12.state_dict(), '../data/model12.pth')

# 保存 model13 的参数
torch.save(model13.state_dict(), '../data/model13.pth')

# 保存 model14 的参数
torch.save(model14.state_dict(), '../data/model14.pth')
# 加载 model1 的参数，并将模型移到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = classify()
model1.load_state_dict(torch.load('../data/model1.pth'))
model1.to(device)

# 加载 model2 的参数，并将模型移到设备上
model2 = classify()
model2.load_state_dict(torch.load('../data/model2.pth'))
model2.to(device)

# 加载 model3 的参数，并将模型移到设备上
model3 = classify()
model3.load_state_dict(torch.load('../data/model3.pth'))
model3.to(device)

# 加载 model4 的参数，并将模型移到设备上
model4 = classify()
model4.load_state_dict(torch.load('../data/model4.pth'))
model4.to(device)

# 加载 model5 的参数，并将模型移到设备上
model5 = classify2()
model5.load_state_dict(torch.load('../data/model5.pth'))
model5.to(device)

# 加载 model6 的参数，并将模型移到设备上
model6 = classify2()
model6.load_state_dict(torch.load('../data/model6.pth'))
model6.to(device)

# 加载 model7 的参数，并将模型移到设备上
model7 = classify2()
model7.load_state_dict(torch.load('../data/model7.pth'))
model7.to(device)

# 加载 model8 的参数，并将模型移到设备上
model8 = classify2()
model8.load_state_dict(torch.load('../data/model8.pth'))
model8.to(device)

# 加载 model9 的参数，并将模型移到设备上
model9 = classify3()
model9.load_state_dict(torch.load('../data/model9.pth'))
model9.to(device)

# 加载 model10 的参数，并将模型移到设备上
model10 = classify3()
model10.load_state_dict(torch.load('../data/model10.pth'))
model10.to(device)

# 加载 model11 的参数，并将模型移到设备上
model11 = classify3()
model11.load_state_dict(torch.load('../data/model11.pth'))
model11.to(device)

# 加载 model12 的参数，并将模型移到设备上
model12 = classify3()
model12.load_state_dict(torch.load('../data/model12.pth'))
model12.to(device)

# 加载 model13 的参数，并将模型移到设备上
model13 = classify4()
model13.load_state_dict(torch.load('../data/model13.pth'))
model13.to(device)

# 加载 model14 的参数，并将模型移到设备上
model14 = classify5()
model14.load_state_dict(torch.load('../data/model14.pth'))
model14.to(device)
class classify_ul(nn.Module):
    def __init__(self):
        super(classify_ul, self).__init__()
        self.fc1 = nn.Linear(42, 3)



    def forward(self, x,y):
        z1=model1(x,y)
        z2=model2(x,y)
        z3=model3(x,y)
        z4=model4(x,y)
        z5=model5(x,y)
        z6=model6(x,y)
        z7=model7(x,y)
        z8=model8(x,y)
        z9=model9(x,y)
        z10=model10(x,y)
        z11=model11(x,y)
        z12=model12(x,y)
        z13=model13(x,y)
        z14=model14(y)
        z=torch.cat((z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14),dim=1)
        z=self.fc1(z)



        return z
torch.cuda.empty_cache()
model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()
model7.eval()
model8.eval()
model9.eval()
model10.eval()
model11.eval()
model12.eval()
model13.eval()
model14.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ul = classify_ul().train().to(device)
train_result_tensor=train_result_tensor.to(device)

optimizer = optim.Adam(model_ul.parameters(), lr=0.0001)
num_epochs = 5
batch_size = 10
torch.cuda.empty_cache()
class_weights = torch.tensor([1.0, 1.5, 0.8], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    running_loss = 0.0

    for id in tqdm(range(4000 // batch_size)):
        optimizer.zero_grad()

        batch_inputs = train_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = train_input_ids[id * batch_size:(id + 1) * batch_size].to(device)

        outputs = model_ul(batch_inputs, batch_labels)
        loss = criterion(outputs, train_result_tensor[id * batch_size:(id + 1) * batch_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        batch_inputs = batch_inputs.to("cpu")
        batch_labels = batch_labels.to("cpu")
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} Loss: {running_loss / (4000 // batch_size)}")
model_ul.eval()
labels=[]
batch_size=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_image_tensor.to(device)
val_input_ids.to(device)
with torch.no_grad():
    for id in tqdm(range(len(val_input_ids) // batch_size)):
        batch_inputs = val_image_tensor[id * batch_size:(id + 1) * batch_size].to(device)
        batch_labels = val_input_ids[id * batch_size:(id + 1) * batch_size].to(device)
        output=model_ul(batch_inputs, batch_labels)
        _, predicted_label = torch.max(output, 1)
        labels.extend(predicted_label)
with open('../test_without_label.txt', "r") as file:
    lines = file.readlines()
for i in range(1, len(lines)):
    line = lines[i].strip()
    label = labels[i-1]

    # 替换"null"为相应的标签
    if label == 0:
        line = line.replace("null", "negative")
    elif label == 1:
        line = line.replace("null", "neutral")
    elif label == 2:
        line = line.replace("null", "positive")

    lines[i] = line

# 将更新后的内容保存到新的txt文件中
with open('../test_with_label.txt', "w") as file:
    file.write(lines[0])
    for line in lines[1:]:
        file.write(line + '\n')