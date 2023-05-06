import torch
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(datatest, batch_size=args.test_bs)
    l = len(dataloader)
    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        log_probs = net_g(data)

        # sum up the batch loss
        test_loss += torch.nn.functional.cross_entropy(log_probs, target, reduction='sum').item()

        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return accuracy, test_loss
