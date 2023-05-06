import pickle
import socket
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from CNN import CNN
from args import argparser


class local_update():
    def __init__(self, args, dataset):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.trainloader = DataLoader(dataset=dataset, batch_size=self.args.client_bs, shuffle=True)


    def train(self, iter, client):
        args = argparser()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CNN(args=args).to(device)

        # Clients load global model parameters
        # net.load_state_dict(torch.load(f"saved_models/Global{iter}.pth"))
        # Clients receive global model parameters from server
        s = socket.socket()
        print("Client: Socket is created.")
        s.connect(("localhost", 9999))
        print("Client connected to the server")
        # received_data = s.recv(1024 * 1024 * 100)
        received_data = b""
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            received_data += packet
        print("Client: Received global model data from server. Size:{}".format(sys.getsizeof(received_data)))
        global_model = pickle.loads(received_data)
        net.load_state_dict(global_model)

        s.close()
        print("Client: Socket is closed.")

        net.train()
        # train and update
        epoch_loss = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for local_iter in range(self.args.client_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(device), labels.to(device)
                net.zero_grad()
                predict = net(images)
                loss = self.loss_func(predict, labels).to(device)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Local Epoch: {:<2d} [{:<4d}/{:<4d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        local_iter, batch_idx * len(images), len(self.trainloader.dataset),
                               100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Clients save local model parameters
        torch.save(net.state_dict(), f'./saved_models/Local_iter{iter}_client{client}.pth')
        return sum(epoch_loss) / len(epoch_loss)