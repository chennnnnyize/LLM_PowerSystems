import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_data(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class ImageClassifier(torch.nn.Module):
    def __init__(self, model_name):
        super(ImageClassifier, self).__init__()
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'densenet':
            self.model = models.densenet121(pretrained=True)
        elif model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
        

        if model_name in ['alexnet', 'vgg16']:
            self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
        elif model_name == 'resnet18':
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        elif model_name == 'densenet':
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 2)
        elif model_name == 'mobilenet':
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)


def train(model, model_name, dataloader, epochs=200, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    best_loss = float('inf')
    epochs_no_improve = 0
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss/(batch_idx+1))

        avg_loss = running_loss / len(dataloader)
        print(f"Average Loss for Epoch {epoch+1}: {avg_loss:.4f}")
        loss_history.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break


    with open('loss_history_{}.txt'.format(model_name), 'w') as f:
        for loss in loss_history:
            f.write(f"{loss}\n")

def test(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True).view_as(target)
            output = output.detach().cpu().numpy().tolist()
            preds = preds.detach().cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(target.detach().cpu().numpy().tolist())


    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)

    dic = {'precision': float(precision), 'recall': float(recall), 
            'f1': float(f1), 'accuracy': float(accuracy)}
    print(dic)

    return dic


data_dirs = ['/root/LLM/GPT4V/train']


model_names = ['alexnet', 'vgg16', 'resnet18', 'densenet', 'mobilenet']
results = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for data_dir in data_dirs:
    dataloader = load_data(data_dir)
    results = {}

    for model_name in model_names:
        model_path = f"models/{model_name}_train.pt"
        print(f"Test {model_name} on {data_dir}")
        model = ImageClassifier(model_name).model.to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()  
            test(model, dataloader)


    # with open('results_{}.json'.format(data_dir), 'w') as f:
    #     json.dump(results, f, indent=4)
