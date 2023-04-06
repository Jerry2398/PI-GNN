import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_enable', action='store', default=True, type=bool)
    parser.add_argument('--dataset_name', action='store', default="cora", type=str)
    parser.add_argument('--save_models_path', action='store', default='save_models', type=str)
    parser.add_argument('--edge_type', action='store', default='stream_edges', type=str)
    parser.add_argument('--logits_name', action='store', default='logits', type=str)
    parser.add_argument('--seed', action='store', default=2021, type=int)
    parser.add_argument('--multi_threading', action='store', default=True, type=bool)

    parser.add_argument('--normalize', action='store', default=False, type=bool)
    parser.add_argument('--m_size', action='store', default=64, type=int)

    parser.add_argument('--init_epochs', action='store', default=400, type=int, help='initial epochs')
    parser.add_argument('--rectify_epochs', action='store', default=10, type=int, help='rectify knowledge epochs')
    parser.add_argument('--retrain_epochs', action='store', default=100, type=int, help='retrain epochs')

    parser.add_argument('--init_patience', action='store', default=20, type=int)
    parser.add_argument('--rectify_patience', action='store', default=4, type=int)
    parser.add_argument('--retrain_patience', action='store', default=4, type=int)

    parser.add_argument('--beta1', action='store', default=0.01, type=float)
    parser.add_argument('--beta2', action='store', default=0.1, type=float)

    parser.add_argument('--init_lr', action='store', default=0.001, type=float)
    parser.add_argument('--rectify_lr', action='store', default=0.001, type=float)
    parser.add_argument('--retrain_lr', action='store', default=0.001, type=float)

    parser.add_argument('--init_weight_decay', action='store', default=0, type=float)
    parser.add_argument('--rectify_weight_decay', action='store', default=0, type=float)
    parser.add_argument('--retrain_weight_decay', action='store', default=0, type=float)
    args = parser.parse_args()
    return args
