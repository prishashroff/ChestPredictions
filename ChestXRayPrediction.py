import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import resnet18
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

#Load the dataset
chest = np.load('chestmnist.npz')

# Commented out IPython magic to ensure Python compatibility.
#Exploratory Data Analysis

#Print these parameters to understand dimensions of dataset
chest.files
chest['train_images'].shape
chest['train_labels'].shape
n_classes = chest['train_labels'].shape[1]

#Visualize the numpy Array as an image
# %matplotlib inline
plt.imshow(chest['train_images'][10], interpolation='nearest')
plt.gray()
plt.show()

#Training Setup- Define Functions

class MyDataset(Dataset):
  def __init__(self, x, y, transform=None):
    super().__init__()
    self.x = x
    self.y = y
    self.transform = transform

  def __getitem__(self, idx):
    img, target = self.x[idx], self.y[idx].astype(int)
    img = Image.fromarray(img)
    img = img.convert('RGB')

    if self.transform is not None:
        img = self.transform(img)
    return img, target

#Resize images to 3x224x224

data_transform = transforms.Compose(
    [transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])])

my_dataset = MyDataset(chest['train_images'], chest['train_labels'], transform=data_transform)
my_dataloader = DataLoader(my_dataset, batch_size=128)

val_dataset = MyDataset(chest['val_images'], chest['val_labels'], transform=data_transform)
val_dataloader = DataLoader(val_dataset, batch_size=128)

test_dataset = MyDataset(chest['test_images'], chest['test_labels'], transform=data_transform)
test_dataloader = DataLoader(test_dataset, batch_size=128)

label_mapping = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural', 'hernia']

#Create the ResNet Model
net = resnet18(pretrained=False, num_classes=14)

#Hyperparameters and Testing Loop
num_epochs = 100
lr = 0.001
gamma=0.1
milestones = [0.5 * num_epochs, 0.75 * num_epochs]

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# Loss function (cross entropy for classification)
loss_func = nn.BCEWithLogitsLoss()

from sklearn.metrics import roc_auc_score, accuracy_score
def test(model, data_loader, criterion, device='cuda', raw=False):
    model.cuda()
    model.eval()

    total_loss = []
    network_answers = []
    true_answers = []
    with torch.no_grad():
        for batch in test_dataloader:
            # Forward pass
            inp, labels = batch
            inp = torch.tensor(inp.cuda(), dtype=torch.float32)
            out = net(inp)

            # Get predictions from scores
            sigmoid = torch.nn.Sigmoid()
            answers = sigmoid(out).data.cpu().numpy()
            preds = np.where(answers >= 0.5, 1, 0)

            # Recording values
            network_answers.extend(preds)
            total_loss.append(loss.item())
            true_answers.extend(labels.data.cpu().numpy())

        auc = roc_auc_score(true_answers, network_answers)
        acc = accuracy_score(true_answers, network_answers)
        testing_loss = np.mean(total_loss)

        if raw:
            return [testing_loss, auc, acc, true_answers, network_answers]

        return [testing_loss, auc, acc]

#Training Loop
net.cuda()

best_epoch = 0
best_auc = 0
best_model = net

for epoch in range(10): # We go over the data ten times
    losses = []
    net.train()
    for batch in my_dataloader:
        optimizer.zero_grad()

        # Forward pass
        inp, labels = batch
        inp = torch.tensor(inp.cuda(), dtype=torch.float32)
        out = net(inp)
        labels = labels.to(torch.float32).cuda()
        loss = loss_func(out, labels)
        losses.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

    train_loss = np.mean(losses)
    val_metrics = test(net, val_dataloader, loss_func)

    cur_auc = val_metrics[1]
    if cur_auc > best_auc:
        best_epoch = epoch
        best_auc = cur_auc
        best_model = net
        print(f"Epoch {best_epoch} is the best yet with Val AUC = {best_auc}")

    scheduler.step()

#Metrics

#Test Loss, Test Accuracy and Test AUC, Classification Report (precision, recall, and f1-score), and a Confusion Matrix

test_loss, test_auc, test_acc, true_answers, network_answers = test(net, test_dataloader, loss_func, raw=True)
print(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}\nTest AUC: {test_auc}")
print(classification_report(true_answers, network_answers, target_names=label_mapping))
print("Multilabel Confusion Matrix:")
print(multilabel_confusion_matrix(true_answers, network_answers))
