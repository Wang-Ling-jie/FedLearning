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
import socket
import pickle
import threading

class LocalTrainThread(threading.Thread):
    def __init__(self, args, dataset, iter, client):
        threading.Thread.__init__(self)
        self.args = args
        self.dataset = dataset
        self.iter = iter
        self.client = client
        self.loss = None
    def run(self) -> None:
        local = local_update(args=self.args, dataset=self.dataset)
        self.loss = local.train(iter=self.iter, client=self.client)


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
    print(f"--------------------------------Global Round {iter+1}----------------------------------")

    w_locals = []
    loss_locals = []
    if args.select != 20:
        selected_clients = random.sample(range(0, 20), args.select)
        print("Partial participation mode clients: \n", [selected_clients[i]+1 for i in range(len(selected_clients))])
    else:
        selected_clients = range(0, 20)

    # server save global model parameters
    # torch.save(net_glob.state_dict(), f'./saved_models/Global{iter}.pth')
    w_glob = net_glob.state_dict()

    for (i, index_client) in enumerate(selected_clients):
        print(f"----------------Training on client {index_client+1} ({i+1} / {len(selected_clients)})----------------")
        LocalTrain = LocalTrainThread(args=args, dataset=train_dataset_clients[index_client], iter=iter, client=index_client)
        LocalTrain.start()
        # Server deliver global model parameters to clients
        # Create TCP socket connection
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("localhost", 9999))
        print("Server socket is bound to an address and port number")
        s.listen(10)
        print("Server is listening for a connection request")

        # Accept connection request
        connected = False
        accept_timeout = 10
        s.settimeout(accept_timeout)
        try:
            connection, address = s.accept()
            print("Server: Connected to a client: {client_info}.".format(client_info=address))
            connected = True
        except socket.timeout:
            print(
                "Server: A socket.timeout exception occurred because the server did not receive any connection for {accept_timeout} seconds.".format(
                    accept_timeout=accept_timeout))

        # Send global model to client
        if connected:
            print("Server: Sending global model to client")
            data = pickle.dumps(w_glob)
            connection.sendall(data)
            print("Server: Global model sent to client. Size: {size}".format(size=len(data)))
            connection.close()
            print("Server: Connection to client closed")

        s.close()
        print("Server socket closed")

        LocalTrain.join()
        loss = LocalTrain.loss
        # Server aggregate local models
        w = torch.load(f'./saved_models/Local_iter{iter}_client{index_client}.pth')
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
plt.savefig('./saved_plt/fed_model{}_epoch{}_select{}.png'.format(args.model, args.epochs, args.select))


# testing
net_glob.eval()
acc_test, loss_test = test_img(net_glob, test_dataset, args)

print("Testing accuracy: {:.2f}".format(acc_test))

