import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import time
import copy


    #train the model on the all the bands
def trainModels(model_list, data_manager, num_epochs, device):

    accuracies = np.zeros(len(model_list))  # we want to keep track of each model's performance

        #for each model
    for index, (model_index, model, optimizer) in enumerate(model_list):

        data_loaders, dataset_sizes = data_manager.getDatasetsForModel(model_index)
        model.to(device)    #move model to CUDA if possible
        model, acc = trainModel(model, optimizer, data_loaders, dataset_sizes, device, num_epochs=num_epochs)
        accuracies[index] = acc.item()  #log model accuracy
        model.to('cpu') #move model back to CPU

    return model_list, accuracies


def trainModel(model, optimizer, data_loaders, dataset_sizes, device, num_epochs, gamma=0.8, step_size=10):
    criterion = nn.CrossEntropyLoss()

    #scheduler to decrease learning rate quickly
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model = runEpochLoop(model, data_loaders, dataset_sizes, device, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs)
    return model


def runEpochLoop(model, data_loaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()

        # we keep track of the best performing model (per validation)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            running_loss, running_corrects = runTrainingLoop(model, data_loaders, criterion, optimizer, phase, device)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return (model, best_acc)


def runTrainingLoop(model, data_loaders, criterion, optimizer, phase, device):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in data_loaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

            # we only enable grad in training
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            # we only update in training
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # stats
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    return running_loss, running_corrects