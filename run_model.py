import torch
from torch import nn
from torch.nn.functional import softmax
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests


def show(img):
    img = img.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    
    
model = models.resnet34(pretrained=True, progress=True)
model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(model.fc.in_features, 8)
)
model.load_state_dict(torch.load("./model_34.pth"))
model.eval()

standardize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

url = "https://image.shutterstock.com/image-photo/crumpled-plastic-blue-bottle-on-260nw-57118336.jpg"
image = Image.open(requests.get(url, stream=True).raw)
input = standardize(image)

predictions = softmax(model(input.unsqueeze(0)), dim=1).data
probability, index = torch.topk(predictions, 1)

labelmapping = {0: 'battery', 1: 'brown-glass', 2: 'cardboard', 3: 'green-glass', 4: 'metal', 5: 'paper', 6: 'plastic', 7: 'white-glass'}
text = f"{labelmapping[index.item()].capitalize()} - {100 * probability.item():.3f}%"

drawing = ImageDraw.Draw(image)
drawing.text((10, 10), text, fill=(0, 0, 0), align='center')

image.show()