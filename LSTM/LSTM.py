import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import sys
import ast
import os
import torchvision.models as models
import tensorflow as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt

def convert_str_to_tensor(tenso_string):
  dict_str = ast.literal_eval(tenso_string)
  tensor_dict = {}
  for key, value in dict_str.items():
    tensor_dict[key] = torch.tensor(value)
  return tensor_dict


def dict_to_list(dictinary):
  lis = []
  for obj in dictinary:
    lis.append(dictinary[obj])
  return lis

class FileContentDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.class_list = os.listdir(directory)
        self.file_list = []

        for class_name in self.class_list:
            class_dir = os.path.join(directory, class_name)
            if os.path.isdir(class_dir):
                files = [(os.path.join(class_name, filename)) for filename in os.listdir(class_dir) if
                         os.path.isfile(os.path.join(class_dir, filename))]
                self.file_list.extend(files)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        relative_path = self.file_list[index]
        #print(relative_path)
        class_name, filename = relative_path.split('/')
        filepath = os.path.join(self.directory, relative_path)

        with open(filepath, 'r') as file:
            content = file.read()
        #print(content)
        #print(filename)
        dict_str = content.replace("\n        ", '')
        dict_str = dict_str.replace("not 3d", '$')
        dict_str = dict_str.replace("3d", '')
        dict_str = dict_str.replace("\n", '')
        dict_str = dict_str.replace("{}", '')
        dict_str = dict_str.replace(", dtype=torch.float64)", '')
        dict_str = dict_str.replace("tensor(", '')
        double_tensor = dict_str.split("$")
        # print(double_tensor)
        tensor_3d = double_tensor[0]
        tensor_reg = double_tensor[1]
        tensor_3d = convert_str_to_tensor(tensor_3d)
        tensor_reg = convert_str_to_tensor(tensor_reg)
        list_3d = dict_to_list(tensor_3d)
        #list_reg = dict_to_list(tensor_reg)
        return list_3d, class_name

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        x = self.fc(h_n[-1])
        return x



def evaluate(model, train_loader,test_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    yona_tot = 0
    yona_corr = 0
    ofir_tot = 0
    ofir_corr = 0
    correct_predictions_test = 0
    total_predictions_test = 0
    with torch.no_grad():
        for frames, labels in train_loader:
            inputs = torch.stack(frames[:], dim=0)
            inputs = inputs.reshape(-1, 5, 17, input_size)  # Reshape input to match LSTM input shape
            for i in range(inputs.size(0)):
                current_tensor = inputs[i]
                if labels[0] == 'ofir':
                    target_labels = torch.tensor([0] * 5)
                else:
                    target_labels = torch.tensor([1] * 5)
                
                current_tensor = current_tensor.to(torch.device('cuda:0'))
                
                outputs = model(current_tensor)
                predicted_labels = torch.argmax(outputs.squeeze(), dim=1)
                correct_predictions += (predicted_labels.to(torch.device('cuda:0')) == target_labels.to(torch.device('cuda:0'))).sum().item()
                total_predictions += target_labels.size(0)

        for frames, labels in test_loader:  # Assuming you have a dataloader for your testing set
            frames = torch.stack(frames[:], dim=0)
            frames = frames.reshape(-1, 5, 17, input_size)
            for i in range(frames.size(0)):
                # Extract the current (5, 17, 3) tensor
                current_tensor = frames[i]
                if labels[0] == 'ofir':
                    target_labels = torch.tensor([0] * 5)
                else:
                    target_labels = torch.tensor([1] * 5)
                
                current_tensor = current_tensor.to(torch.device('cuda:0'))
                
                test_out = model(current_tensor)
                predicted_labels = torch.argmax(test_out.squeeze(), dim=1)

                # Compare the predicted labels with the target labels
                predicted_labels = predicted_labels.to(torch.device('cuda:0'))
                target_labels = target_labels.to(torch.device('cuda:0'))
                correct_predictions_test += (predicted_labels == target_labels).sum().item()
                
                if labels[0] == 'ofir':
                    ofir_corr += (predicted_labels == target_labels).sum().item()
                    ofir_tot += target_labels.size(0)
                else:
                    yona_corr += (predicted_labels == target_labels).sum().item()
                    yona_tot += target_labels.size(0)
                # accuracy = correct_predictions / target_labels.size(0)
                total_predictions_test += target_labels.size(0)
    model.train()
    accuracy_train = correct_predictions / total_predictions
    accuracy_test = correct_predictions_test / total_predictions_test
    accuracy_yona = yona_corr/yona_tot
    accuracy_ofir = ofir_corr/ofir_tot

    return [accuracy_train,accuracy_test,accuracy_yona,accuracy_ofir]





BATCH_SIZE = 1
input_size = 3
hidden_size = 128
num_classes = 2
learning_rate = 0.001
num_epochs = 250
class_labels = ['yonatan', 'ofir']

model = RNNClassifier(input_size, hidden_size, num_classes)
model = model.to(torch.device('cuda:0'))
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


path = "/home/ofiryonatan/Neural_Network/data"
data = FileContentDataset(path)

dataloader = DataLoader(data, batch_size=1, shuffle=False)

train_subset, test_subset = torch.utils.data.random_split(data, [73, 19])#the data here is the costume dataset
train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_subset, shuffle=False, batch_size=BATCH_SIZE)
acc_train = {}
acc_test = {}
acc_yona = {}
acc_ofir = {}

lst_acc_train = []
lst_acc_test = []
lst_acc_yona = []
lst_acc_ofir = []

train_loss = []
test_loss = []
epoch_loss = 0
normalized_loss = 0
total_steps =0


for epoch in range(num_epochs):
    j = 0
    epoch_loss = 0
    for frames, labels in train_loader:  # Assuming you have a dataloader for your training set
        combined_tensor_list = []
        frames = torch.stack(frames[:], dim=0)
        frames = frames.reshape(-1, 5,17, input_size)  # Reshape input to match LSTM input shape
        loss = 0
        for i in range(frames.size(0)):
            current_tensor = frames[i]
            if labels[0] == 'ofir':
                target_labels = torch.tensor([0]*5)
            else:
                target_labels = torch.tensor([1]*5)
            
            current_tensor = current_tensor.to(torch.device('cuda:0'))
            
            target_one_hot = (F.one_hot(target_labels, num_classes=2).float()).to(torch.device('cuda:0'))
            outputs = model(current_tensor)
            #loss = loss_fn(outputs.squeeze(), target_labels)
            loss = loss_fn(outputs.squeeze(), target_one_hot)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j += 1
        total_steps += 1
        # Print training loss for every few iterations
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{j + 1}/{73*10}], Loss: {loss.item():.4f}')
    if(len(train_loss) == 0):
      normalized_loss = epoch_loss.item()
    train_loss.append(epoch_loss.item()/normalized_loss)
    accurecies = evaluate(model,train_loader,test_loader)
    acc_train[str(epoch)] = accurecies[0]
    acc_test[str(epoch)] = accurecies[1]
    acc_yona[str(epoch)] = accurecies[2]
    acc_ofir[str(epoch)] = accurecies[3]
    lst_acc_train.append(accurecies[0])
    lst_acc_test.append(accurecies[1])
    lst_acc_yona.append(accurecies[2])
    lst_acc_ofir.append(accurecies[3])
    print()
    print("epoch number " +str(epoch+1))
    print("the accurecy of the train was: " +str(accurecies[0]))
    print("the accurecy of the test was: " + str(accurecies[1]))
    print("the accurecy of the yona was: " + str(accurecies[2]))
    print("the accurecy of the ofir was: " + str(accurecies[3]))
    save_path = "/home/ofiryonatan/Neural_Network/weights/rnn_model_weights"+str(epoch+1)+".pth"
    torch.save(model.state_dict(), save_path)
    #print("the accurecy on the train was: "+str(evaluate(model,train_loader,test_loader)))
        #if (i+1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Normalized Train Loss per Epoch')
plt.show()

"""
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), lst_acc_train, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), lst_acc_test, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Vs. Test Accuracy per Epoch')
plt.show()
"""


plt.figure(figsize=(10, 6))
#plt.plot(range(1, num_epochs+1), lst_acc_train, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), lst_acc_test, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy per Epoch')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), lst_acc_yona, label='Yonatan Accuracy')
plt.plot(range(1, num_epochs+1), lst_acc_ofir, label='Ofir Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Yonatan Vs. Ofir Accuracy per Epoch')
plt.show()


model.eval()  # Switch to evaluation mode
with torch.no_grad():
    correct_predictions = 0
    total = 0
    yona_tot = 0
    yona_corr = 0
    ofir_tot = 0
    ofir_corr = 0
    j = 1
    for frames, labels in test_loader:  # Assuming you have a dataloader for your testing set
        frames = torch.stack(frames[:], dim=0)
        frames = frames.reshape(-1, 5, 17, input_size)
        for i in range(frames.size(0)):
            # Extract the current (5, 17, 3) tensor
            current_tensor = frames[i]
            if labels[0] == 'ofir':
                target_labels = torch.tensor([0]*5)
            else:
                target_labels = torch.tensor([1]*5)
            
            current_tensor = current_tensor.to(torch.device('cuda:0'))
            
            test_out = model(current_tensor)
            predicted_labels = torch.argmax(test_out.squeeze(), dim=1)

            # Compare the predicted labels with the target labels
            predicted_labels = predicted_labels.to(torch.device('cuda:0'))
            target_labels = target_labels.to(torch.device('cuda:0'))
            correct_predictions += (predicted_labels == target_labels).sum().item()
            if labels[0] == 'ofir':
                ofir_corr += (predicted_labels == target_labels).sum().item()
                ofir_tot+=target_labels.size(0)
            else:
                yona_corr += (predicted_labels == target_labels).sum().item()
                yona_tot += target_labels.size(0)
            #accuracy = correct_predictions / target_labels.size(0)
            total += target_labels.size(0)

            # Print the evaluation results
        accuracy = correct_predictions / total
        print("Accuracy after file number "+str(j)+": " + str(accuracy))
        j+=1
            #predicted_labels = torch.argmax(test_out.squeeze(), dim=1)
    save_path = "rnn_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print("Model weights saved successfully.")
    print("yonatan's accurecy is: "+str(yona_corr/yona_tot))
    print("ofir's accurecy is: " + str(ofir_corr / ofir_tot))
    print("all accurecies: ")
    print(acc_train)
    print("--------------")
    print(acc_test)
    print("--------------")
    print(acc_yona)
    print("--------------")
    print(acc_ofir)
    if j > 1:
        print("done")











#model = FullyConnectedModel()
#model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#for RESNET
"""resnet = models.resnet18(pretrained=True)
num_features = resnet.fc.in_features
num_channels = resnet.conv1.in_channels
resnet.fc = Classifier(num_features* 5 * 17, num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
# Define the loss function and optimizer"""

""""#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
initial_learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
"""

# input_data = current_tensor.unsqueeze(0)
# target_indices = [class_labels.index(label) for label in labels]
# target_tensor = torch.LongTensor(target_indices[0])
"""label_mapping = {label: index for index, label in enumerate(labels)}
encoded_labels = [label_mapping[label] for label in labels]"""
# target_labels_tensor = torch.tensor(encoded_labels)
# current_tensor = current_tensor.unsqueeze(0)