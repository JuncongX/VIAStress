import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='SAGM')
parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys'])
parser.add_argument('--y_dim', type=int, default=2)
parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])
parser.add_argument('--k', type=int, default=5, choices=[5, 7])
parser.add_argument('--context_num', type=int, default=5)

parser.add_argument('--r_dim', type=int, default=256)
parser.add_argument('--z_dim', type=int, default=256)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--x_dim', type=int, default=128)

parser.add_argument("--stop_gradient", type=bool, default=False,
                    help='whether stop gradient of the first order gradient')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--fps', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step_size', type=float, default=1)
parser.add_argument('--inner_loops', type=int, default=200)
parser.add_argument('--pre_loops', type=int, default=0)
parser.add_argument('--test_every', type=str2bool, default=True)
parser.add_argument('--debug', type=str2bool, default=True)
parser.add_argument('--state_dict', type=str, default="")
parser.add_argument('--model_path', type=str, default="./checkpoints")
parser.add_argument('--cutoff_low', type=float, default=0.5)
parser.add_argument('--cutoff_high', type=float, default=8.0)
parser.add_argument('--bwf_order', type=int, default=3)
parser.add_argument('--meta_step_size', type=float, default=0.001)
parser.add_argument('--meta_loss_scale', type=float, default=0.1)
parser.add_argument('--meta_val_beta', type=float, default=0.1)
parser.add_argument('--tta', type=str2bool, default=True, help="using testing-time adaptation or not")
parser.add_argument('--gdl', type=str2bool, default=True, help="using gradient direction loss or not")
parser.add_argument('--upt', type=str2bool, default=True, help="using unsupervised pretext task or not")
args = parser.parse_args()
