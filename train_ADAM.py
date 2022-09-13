import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import model as model_lib
import dataset as dataset_lib

def test(model,loaders):
    """
    Testing a model on the testset

    :param model: the pytorch model
    :param loaders: dataset loaders
    :return: the average accuracy of the model evaluated on teh complete test set
    """
    model.eval()
    with torch.no_grad():
        nr_of_tested_batches = 0
        sum=0
        for images, labels in loaders['test']:
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            sum+=accuracy
            nr_of_tested_batches+=1
    return sum/nr_of_tested_batches

def gradient_based_train(num_epochs, cnn, loaders,optimizer,loss_func):
    """
    Training a model with ADAM gradient based optimiser
    This function is meant to be a baseline that can be used in order to compare how well GA training performs in comparision to ADAM

    :param num_epochs: how many epochs to train for
    :param cnn: the model that should be trained
    :param loaders:  loaders to the training and testset
    :param optimizer: the optimizer to use (ADAM)
    :param loss_func: loss function to use (CrossEntropyLoss)
    :return: None
    """
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # clear gradients for this training step
            optimizer.zero_grad()
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y

            output = cnn(b_x)
            loss = loss_func(output, b_y)

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                accuracy = test(cnn, loaders)
                print("testset accuracy:"+str(accuracy))


if __name__ == "__main__":
    cnn = model_lib.LeNet5(n_classes=10)
    gradient_based_train(num_epochs=500,cnn=cnn,loaders=dataset_lib.get_loaders(train_batch_size=100),loss_func=nn.CrossEntropyLoss(),optimizer = optim.Adam(cnn.parameters(), lr=0.01))

