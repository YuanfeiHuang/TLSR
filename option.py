import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description='TLSR')

# Hardware specifications
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda?')
parser.add_argument('--n_GPUs', type=int, default=1, help='parallel training with multiple GPUs')
parser.add_argument('--GPU_ID', type=str, default=0, help='GPUs id')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data specifications
# parser.add_argument('--dir_data', type=str, default='data/Datasets/', help='dataset directory')
parser.add_argument('--dir_data', type=str, default='../../Datasets/', help='dataset directory')
parser.add_argument('--data_train', type=str, default='DF2K', help='train dataset name')
# parser.add_argument('--data_test', type=str, default=['Set5', 'Set14', 'BSD100', 'Urban100'], help='validation/test dataset')
parser.add_argument('--data_test', type=str, default=['Set14'], help='validation/test dataset')
# parser.add_argument('--data_test', type=str, default=['classic5', 'LIVE1'], help='validation/test dataset')
parser.add_argument('--n_train', type=int, default=800, help='number of training set')
parser.add_argument('--shuffle', type=bool, default=False, help='')
parser.add_argument("--store_in_ram", default=True, action="store_true", help="Use cuda?")
# parser.add_argument('--degrad_train', type=float, default={'type': 'JPEG', 'min_sigma': 10, 'max_sigma': 30},
#                   help='degradation settings for training')
parser.add_argument('--blur_size', type=int, default=21, help='number of training set')
# parser.add_argument('--degrad_train', type=float, default={'type': 'B', 'min_sigma': 0.2, 'max_sigma': 4.0},
#                    help='degradation settings for training')
# parser.add_argument('--degrad_train', type=float, default={'type': 'B_aniso', 'min_sigma': 0.0, 'max_sigma': 0.5},
#                     help='degradation settings for training')
parser.add_argument('--degrad_train', type=float, default={'type': 'N', 'min_sigma': 0.0, 'max_sigma': 30.0},
                 help='degradation settings for training, type B for convolutive degradations, type N for additive degradations')
# parser.add_argument('--degrad_test', type=float, default=[{'type': 'JPEG', 'sigma': 10},
#                                                          {'type': 'JPEG', 'sigma': 20},
#                                                          {'type': 'JPEG', 'sigma': 30}],
#                    help='degradation settings for testing/validation type B for convolutive degradations, type N for additive degradations')
# parser.add_argument('--degrad_test', type=float, default=[{'type': 'B', 'sigma': 0.2},
#                                                            {'type': 'B', 'sigma': 1.0},
#                                                            {'type': 'B', 'sigma': 2.0},
#                                                            {'type': 'B', 'sigma':  3.0},
#                                                           {'type': 'B', 'sigma': 4.0}],
#                    help='degradation settings for testing/validation')
# parser.add_argument('--degrad_test', type=float, default=[{'type': 'B_aniso', 'sigma': 0.0},
#                                                           {'type': 'B_aniso', 'sigma': 0.167},
#                                                           {'type': 'B_aniso', 'sigma': 0.25},
#                                                           {'type': 'B_aniso', 'sigma': 0.333},
#                                                           {'type': 'B_aniso', 'sigma': 0.5}],
#                    help='degradation settings for testing/validation type B for convolutive degradations, type N for additive degradations')
parser.add_argument('--degrad_test', type=float, default=[{'type': 'N', 'sigma': 0.0},
                                                          # {'type': 'N', 'sigma': 5.0},
                                                          # {'type': 'N', 'sigma': 10.0},
                                                          {'type': 'N', 'sigma': 15.0},
                                                          # {'type': 'N', 'sigma': 20.0},
                                                          # {'type': 'N', 'sigma': 25.0},
                                                          {'type': 'N', 'sigma': 30.0}],
                 help='degradation settings for testing/validation')
parser.add_argument('--model_path', type=str, default='', help='path to save model')
parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48, help='output patch size')
parser.add_argument('--value_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--chop_forward', action='store_true', help='enable memory-efficient forward')

# Model specifications:
parser.add_argument('--n_homo_blocks', type=int, default=[4, 8], help='number of blocks in homo feature extraction module')
parser.add_argument('--n_transi_layers', type=int, default=[1, 8], help='number of layers in transitive learning module')
parser.add_argument('--n_channels', type=int, default=128, help='number of feature channels')
parser.add_argument('--n_homo_width', type=int, default=64, help='number of layers in transitive learning module')
parser.add_argument('--n_transi_width', type=int, default=64, help='number of layers in transitive learning module')
parser.add_argument('--act', default=nn.ReLU(True), help='activation function')
parser.add_argument('--num_samples', type=int, default=8, help='number of samples in random cropping module')
parser.add_argument('--size_samples', type=int, default=32, help='spatial size of samples in random cropping module')

# Training/Testing specifications
parser.add_argument('--train', type=str, default='Test', help='True for training, False for testing')
parser.add_argument('--iter_epoch', type=int, default=2000, help='iteration in each epoch')
parser.add_argument('--start_epoch_DoT', default=0, type=int, help='start epoch for training')
parser.add_argument('--start_epoch_SR', default=0, type=int, help='start epoch for training')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--resume_SR', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--resume_DoT', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--run_DoT', default=True, help='train DoTNet?')
parser.add_argument('--DoT', type=str, default='est', help='use GT DoT or Estimated DoT?')
parser.add_argument('--run_SR', default=True, help='train SR_Model?')

# Optimization specifications
parser.add_argument('--lr', type=float, default={'SR': 0.0002, 'DoTNet': 0.0008}, help='initial learning rate')
parser.add_argument('--lr_step_size', type=int, default={'SR': 50, 'DoTNet': 50}, help='learning rate decay per N epochs')
parser.add_argument('--lr_gamma', type=int, default={'SR': 0.5, 'DoTNet': 0.5}, help='learning rate decay factor for step decay')

args = parser.parse_args()
