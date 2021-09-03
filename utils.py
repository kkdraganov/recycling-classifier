import json
import os
from PIL import Image
import random
from urllib.request import urlretrieve as save_file


def load_json(filepath):
    with open(filepath, 'r') as file:
        json_text = file.read()
        data = json.loads(json_text)
        return data

def save_files(source_json, destination):
    for image in source_json['images']:
        image_name = image['file_name']
        url = image['coco_url']
        save_file(url, destination + image_name)

def crop(source_json, destination, image_type):
    mapping = {1: 'glass', 2: 'metal', 3: 'plastic'}
    count = {'glass': 0, 'metal': 0, 'plastic': 0}
    
    for annotation in source_json['annotations']:
        image_data = source_json['images'][annotation['image_id']]
        
        image_name = image_data['file_name']
        category = mapping[annotation['category_id']]
        
        image = Image.open(f"./data/images/full_images/{image_type}/{image_name}")
        
        b = annotation['bbox']
        box = (b[0], b[1], b[0] + b[2], b[1] + b[3])
        image = image.crop(box)
        
        count[category] += 1
        image.save(f"{destination}/{category}/{category}{count[category]:03d}.png", "png")

def train_test_split(base_directory="./data/images/zArchive/all/", training_directory="./data/images/training/", validation_directory="./data/images/validation/", testing_directory="./data/images/testing/"):
    subfolders = os.listdir(base_directory)
    print(subfolders)
        
    for subfolder in subfolders:
        path = base_directory + subfolder + '/'
        images = os.listdir(path)
        
        # Get images that will be for validation and testing purposes
        length = int(0.1 * len(images))
        random_indexes = random.sample(range(len(images)), 2 * length)
        valid_indexes, testing_indexes = random_indexes[:length], random_indexes[length:]
    
        for i in range(len(images)):
            if i in testing_indexes:
                destination = testing_directory + subfolder
            elif i in valid_indexes:
                destination = validation_directory + subfolder
            else:
                destination = training_directory + subfolder
                
            image = Image.open(path + images[i])
            image.save(f"{destination}/{subfolder}{i+1:03d}.png", "png")