# coding: utf-8
import argparse
import time
import math
import os

import torch
import torch.optim as optim


def train(trainloader, testloader, model, nepochs, optimizer, criterion, device="cpu"):
    model.to(device)

    for epoch in range(nepochs):

        model = model.train()
        running_loss = 0.0

        dgates = [ i for i in model.children() if hasattr(i, "get_sparsity_loss") ]

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            for gate in dgates:
                loss += gate.reg_weight * gate.get_sparsity_loss()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0  
                str_ = ""
                for gate in dgates:
                    str_ += str(float(gate.get_sparsity())) + "\t"
                print(str_)

        result = evaluate(testloader, model, device)
        print('Accuracy: %.2f %%' % (result)) 


def evaluate(testloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total
