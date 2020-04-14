import configparser
import torchvision.datasets as dset
import torchvision.transforms as transforms
from network import SiameseNetworkDataset, SiameseNetwork, ContrastiveLoss
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch
from util import imshow, show_plot
from torch import optim
import torch.nn.functional as F

def training_process(net, dataset):
    train_dataloader = DataLoader(dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=int(config.get("para", "train_batch_size")))
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=float(config.get("para", "learning_rate")))

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, int(config.get("para", "train_number_epochs"))):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter, loss_history)
    torch.save(net, "siameseNet.model")
    return net

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read_file(open('config.ini'))

    folder_dataset = dset.ImageFolder(root=config.get("dir", "training_dir"))
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)
    #
    # vis_dataloader = DataLoader(siamese_dataset,
    #                             shuffle=True,
    #                             num_workers=0,
    #                             batch_size=8)
    # dataiter = iter(vis_dataloader)
    #
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    #
    # train_dataloader = DataLoader(siamese_dataset,
    #                               shuffle=True,
    #                               num_workers=0,
    #                               batch_size=int(config.get("para", "train_batch_size")))
    #net = SiameseNetwork()
    net = torch.load("siameseNet.model")
    #net = training_process(net, siamese_dataset)

    #net = torch.load("siameseNet.model")

    folder_dataset_test = dset.ImageFolder(root=config.get("dir", "testing_dir"))
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=False)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(12):
        _, x1, _ = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = net(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        # similarity = 20/(euclidean_distance.item()+0.035)
        # if similarity > 100:
        #     similarity = 100.0
        # imshow(torchvision.utils.make_grid(concatenated), 'Similarity: {:.2f}%'.format(similarity))
        imshow(torchvision.utils.make_grid(concatenated), 'Similarity: {:.2f}'.format(euclidean_distance.item()))

