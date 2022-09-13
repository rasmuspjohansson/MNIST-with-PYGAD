import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def get_train_data(transforms):

    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = transforms,
        download = True,
    )
    return train_data
def get_test_data(transforms):
    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = transforms
    )
    return test_data



#create loader

def get_loaders(train_batch_size):
    # define transforms
    mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])
    train_data =get_train_data(transforms=mnist_transforms)
    test_data= get_test_data(transforms=mnist_transforms)
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=1),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=1024,
                                            shuffle=True,
                                            num_workers=1),
    }
    return loaders

if __name__ == "__main__":
    print("creating train and validationset loaders")
    loaders = get_loaders(train_batch_size=8)
    print("creating an iterator for the trainingset")
    train_loaders_iter = iter(loaders["train"])
    print("loading the first batch")
    (data_inputs, data_outputs) = next(train_loaders_iter)
    print("show first image in the batch")
    im_nr=0
    plt.imshow(data_inputs.data[im_nr].squeeze(), cmap='gray')
    plt.title('%i' % data_outputs[im_nr])
    plt.show()
