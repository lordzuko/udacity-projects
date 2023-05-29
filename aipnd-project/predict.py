# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import numpy as np
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
        size[0]//2 - 112,
        size[1]//2 - 112,
        size[0]//2 + 112,
        size[1]//2 + 112)
    )
    np_image = np.array(image)
    #Scale Image per channel
    # Using (image-min)/(max-min)
    np_image = np_image/255.
        
    img_a = np_image[:,:,0]
    img_b = np_image[:,:,1]
    img_c = np_image[:,:,2]
    
    # Normalize image per channel
    img_a = (img_a - 0.485)/(0.229) 
    img_b = (img_b - 0.456)/(0.224)
    img_c = (img_c - 0.406)/(0.225)
        
    np_image[:,:,0] = img_a
    np_image[:,:,1] = img_b
    np_image[:,:,2] = img_c
    
    # Transpose image
    np_image = np.transpose(np_image, (2,0,1))
    return np_image


#function that loads a checkpoint and rebuilds the model
def load_checkpoint(args):
    checkpoint = torch.load(args.saved_model_path)
    if checkpoint['arch'] == 'densenet':
        model = models.densenet121()
    elif checkpoint['arch'] == 'vgg':
        model = models.vgg16()

    # build the classifier part of model
    classier_net = []
    hidden_units = [int(x) for x in checkpoint['hidden_units'].split(',')]
    hidden_units = [1024] + hidden_units
    hidden_units_pair = list(zip(hidden_units,hidden_units[1:]))
    hidden_units_pair.append((hidden_units[-1], checkpoint['output_size']))
    for i,x in enumerate(hidden_units_pair):     
        classier_net.append(('fc'+str(i+1), nn.Linear(*(hidden_units_pair[i]))))
        

    classier_net.append(('output', nn.LogSoftmax(dim=1)))


    classifier =  nn.Sequential(OrderedDict(classier_net))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    # Check whether to train on gpu or not
    if args.gpu:
        # If gpu is available use gpu
        if torch.cuda.is_available():
            model.cuda()
        else:
            print('GPU not available, continuing prediction on cpu')

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class

def predict(image_path, model, class_to_idx, idx_to_class ,cat_to_name,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.FloatTensor([image])
    # set model in evaluation mode
    model.eval()
    output = model.forward(Variable(image))
    ps = torch.exp(output).data.numpy()[0]

    topk_index = np.argsort(ps)[-topk:][::-1] 
    topk_class = [idx_to_class[x] for x in topk_index]
    named_topk_class = [cat_to_name[x] for x in topk_class]
    topk_prob = ps[topk_index]

    return topk_prob, named_topk_class

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--image_path', type=str, help='path of image to be predicted')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

    args = parser.parse_args()


    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    model, class_to_idx, idx_to_class = load_checkpoint(args)
    topk_prob, named_topk_class = predict(args.image_path, model, 
                                          class_to_idx, 
                                          idx_to_class, 
                                          cat_to_name, 
                                          topk=args.topk)
                                          
    print('Classes: ', named_topk_class)
    print('Probability: ', topk_prob)

if __name__ == "__main__":
    main()