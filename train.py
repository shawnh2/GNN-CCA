import argparse

from tools.trainer import train


def make_train_parser():
    parser = argparse.ArgumentParser("GNN CCA train parser")
    # training config
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="batch size for validating")
    parser.add_argument("--device", type=str, default="cuda", help="device for training")
    parser.add_argument("--output", type=str, default="./output", help="output dir for training")
    parser.add_argument("-s", "--max-passing-steps", type=int, default=4,
                        help="maximum message passing steps in GNN for MPN")
    # reid feature extractor
    parser.add_argument("--reid-name", type=str, default="osnet_ain_x1_0",
                        help="the name of feature extractor model")
    parser.add_argument("--reid-path", type=str, default="ckpt/osnet_ain_ms_d_c.pth.tar",
                        help="the path to feature extractor model")
    # dataset
    parser.add_argument("--epfl", default=False, action="store_true",
                        help="using EPFL dataset for training")
    parser.add_argument("--seq-name", type=str, nargs="+",
                        help="using specific sequences in dataset, 'all' means using all of them")

    return parser


if __name__ == '__main__':
    args = make_train_parser().parse_args()
    train(args)
