import torch
from torch import nn
from torch.nn.functional import softmax
from torch import optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

import time
from datetime import timedelta
import matplotlib.pyplot as plt


def get_model(pretrained=False):
    model = models.resnet34(pretrained=pretrained)

    # Append an additional layer with neurons for each of the 8 classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(num_features, 8)
    )
    return model


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation((0, 180)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_images = datasets.ImageFolder("./data/images/training", train_transform)
    training_loader = DataLoader(training_images, batch_size=25, shuffle=True, num_workers=2)
    
    validation_images = datasets.ImageFolder("./data/images/validation", valid_transform)
    validation_loader = DataLoader(validation_images, batch_size=25, shuffle=False, num_workers=2)
    
    testing_images = datasets.ImageFolder("./data/images/testing", test_transform)
    testing_loader = DataLoader(testing_images, batch_size=25, shuffle=False, num_workers=2)
    
    dataloaders = {'train': training_loader, 'valid': validation_loader, 'test': testing_loader}
    return dataloaders


def train_model(model, device, dataloaders, optimizer, criterion, epoch_count=25):
    training_start = time.time()
    
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
        
    train_losses = []
    valid_losses = []

    for epoch in range(1, epoch_count + 1):
        train_loss = 0.0
        valid_loss = 0.0
        
        ''' Training '''
        # Set model to training
        model.train()
        
        for images, labels in train_loader:
            # Move tensors to correct device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Run the model and calculate the losses
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backpropagate the losses
            loss.backward()
            optimizer.step()
            
            # Save the losses
            train_loss += loss.item() * images.size(0)
          
            
        ''' Validation '''
        # Set the model to evaluation
        model.eval()
            
        for images, labels in valid_loader:
            # Move tensors to correct device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Run the model and calculate the losses
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Save the losses
            valid_loss += loss.item() * images.size(0)
            
        
        ''' Store losses '''
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f"Epoch: {epoch} \tTraining loss: {train_loss:.6f} \tValidation loss: {valid_loss:.6f}")

    print(f"\nTraining time: {timedelta(seconds = time.time() - training_start)}")
    return model, train_losses, valid_losses
    

if __name__ == "__main__":
    # labelmapping = {0: 'battery', 1: 'brown-glass', 2: 'cardboard', 3: 'green-glass', 4: 'metal', 5: 'paper', 6: 'plastic', 7: 'white-glass'}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    ''' Load and train model '''
    base_model = get_model(True)
    # base_model.load_state_dict(torch.load("./model_34.pth"))
    base_model = base_model.to(device)
    
    dataloaders = get_dataloaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    
    model, t_loss, v_loss = train_model(base_model, device, dataloaders, optimizer, criterion, 50)
    
    
    ''' Calculate accuracy '''
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloaders['test']:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted = torch.max(softmax(outputs.data, dim=1), 1)[1]
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
              
        print(f"Model accuracy: {100 * correct / total:.1f}%")
    
    
    ''' Plot losses '''
    plt.plot(t_loss, label='Training loss')
    plt.plot(v_loss, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    
    
    ''' Save model '''
    torch.save(model.state_dict(), "./model_34.pth")
