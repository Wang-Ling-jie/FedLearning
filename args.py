import argparse

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--select', type=int, default=20, help='the number of selected clients: C')
    parser.add_argument('--client_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--client_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--test_bs', type=int, default=64, help="test batch size")

    return parser.parse_args()