import copy
import socket
import dill
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from args import argparser
import random
import args
from CNN import CNN
from local_update import local_update
from test import test_img


train_dataset_clients = []
for i in range(20):
    with open(f"Data/Client{i+1}.pkl",'rb') as f:
        train_dataset_clients.append(dill.load(f))


def FedAvg(w_locals):
    w_avg = {}
    for k in w_locals[0].keys():
        w_avg[k] = torch.zeros(w_locals[0][k].shape).to(device)
        for i in range(len(w_locals)):
            w_avg[k] += w_locals[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg


transform = transforms.Compose([
    transforms.ToTensor(),
])


test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

print(test_dataset)
args = argparser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_glob = CNN(args=args).to(device)
net_glob.train()
w_glob = net_glob.state_dict()

# training
lr = 1e-4
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
optimizer = torch.optim.Adam(net_glob.parameters(), lr=lr)


for iter in range(args.epochs):
    print(f"------------------------Global Round {iter+1}--------------------------")

    w_locals = []
    loss_locals = []
    if args.select != 20:
        selected_clients = random.sample(range(0, 20), args.select)
        print("Partial participation mode clients: ", [selected_clients[i]+1 for i in range(len(selected_clients))])
    else:
        selected_clients = range(0, 20)

    # server save global model parameters
    torch.save(net_glob.state_dict(), f'./saved_models/Global{iter}.pth')

    for index_clients in selected_clients:
        print(f"----------Training on client {index_clients+1}----------")
        local = local_update(args=args, dataset=train_dataset_clients[i])
        loss = local.train(iter=iter, client=index_clients)
        # Server aggregate local models
        w = torch.load(f'./saved_models/Local_iter{iter}_client{index_clients}.pth')
        w_locals.append(w)
        loss_locals.append(copy.deepcopy(loss))
    # Update global weights
    w_glob = FedAvg(w_locals)

    # Load weight to global model
    net_glob.load_state_dict(w_glob)

    # Print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter+1, loss_avg))
    loss_train.append(loss_avg)


# Draw loss figure
plt.figure()
plt.plot(range(len(loss_train)), loss_train)
plt.ylabel('train_loss')
plt.savefig('./save_plt/fed_model{}_epoch{}_select{}.png'.format(args.model, args.epochs, args.select))


# testing
net_glob.eval()
acc_test, loss_test = test_img(net_glob, test_dataset, args)

print("Testing accuracy: {:.2f}".format(acc_test))

