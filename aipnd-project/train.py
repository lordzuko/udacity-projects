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
import time


def prepare_dataloaders(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=5)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=5)

    return trainloader, validloader, testloader

def train_network(args):
    if args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
    elif args.arch == 'vgg':
        model = models.vgg16(pretrained=True)
    
    # Freeze the parameters of densenet so that losses doen't back propagate
    for param in model.parameters():
        param.requires_grad = False
    
    # Build and train your network
    classier_net = []
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
    hidden_units = [1024] + hidden_units
    hidden_units_pair = list(zip(hidden_units,hidden_units[1:]))
    hidden_units_pair.append((hidden_units[-1], 102))
    for i,x in enumerate(hidden_units_pair):     
        classier_net.append(('fc'+str(i+1), nn.Linear(*(hidden_units_pair[i]))))
        

    classier_net.append(('output', nn.LogSoftmax(dim=1)))

    classifier =  nn.Sequential(OrderedDict(classier_net))
    model.classifier = classifier

    print(model)
    
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    # Check whether to train on gpu or not
    if args.gpu:
        # If gpu is available use gpu
        if torch.cuda.is_available():
            model.cuda()
        else:
            print('GPU not available, continuing training on cpu')
    
    trainloader, validloader, testloader = prepare_dataloaders(args)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 1
    start = time.time()
    try:
        for e in range(epochs):
            for images, labels in iter(trainloader):
                steps += 1
        #         images.resize_(images.size()[0], 3*224*224)
                
                # Wrap images and labels in Variables so we can calculate gradients
                inputs = Variable(images)
                targets = Variable(labels)
                optimizer.zero_grad()
                
                output = model.forward(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.data[0]
                
                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()
                    
                    accuracy = 0
                    val_loss = 0
                    for ii, (images, labels) in enumerate(validloader):
        #                 images = images.resize_(images.size()[0], 3*224*224)
                        # Set volatile to True so we don't save the history
                        inputs = Variable(images, volatile=True)
                        labels = Variable(labels, volatile=True)

                        output = model.forward(inputs)
                        val_loss += loss_fn(output, labels).data[0]
                        
                        ## Calculating the accuracy 
                        # Model's output is log-softmax, take exponential to get the probabilities
                        ps = torch.exp(output).data
                        # Class with highest probability is our predicted class, compare with true label
                        equality = (labels.data == ps.max(1)[1])
                        # Accuracy is number of correct predictions divided by all predictions, just take the mean
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(val_loss/len(validloader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                    
                    running_loss = 0
                    
                    # Make sure dropout is on for training
                    model.train()
    except KeyboardInterrupt:
        pass

    run_time = time.time() - start
    print('Training completed in {:.0f}m and {:.0f}s'.format(
        run_time // 60, run_time % 60))

    print('Starting Validation on Test Set')
    #validation on the test set
    model.eval()
                
    accuracy = 0
    test_loss = 0
    for ii, (images, labels) in enumerate(testloader):
        # Set volatile to True so we don't save the history
        inputs = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += loss_fn(output, labels).data[0]

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output).data
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

    
    # Save the model
    checkpoint = {'input_size': [3, 224, 224],
                'batch_size': trainloader.batch_size,
                'output_size': 102,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer_dict':optimizer.state_dict(),
                'class_to_idx':  trainloader.dataset.class_to_idx,
                'lr': args.lr,
                'hidden_units': args.hidden_units,
                'epoch': epochs}
    print('Saving Model')
    torch.save(checkpoint, args.saved_model_path)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--arch', type=str, default='densenet', help='architecture [available: densenet, vgg]', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=str, default='500', help='hidden units for fc layers (comma separated)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
    args = parser.parse_args()

    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    train_network(args)


if __name__ == "__main__":
    main()