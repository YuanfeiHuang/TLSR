import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description='TLSR')

# Hardware specifications
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda?')
parser.add_argument('--n_GPUs', type=int, default=1, help='parallel training with multiple GPUs')
parser.add_argument('--GPU_ID', type=int, default=0, help='GPUs id')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data specifications
parser.add_argument('--dir_data', type=str, default='data/Datasets/', help='dataset directory')
parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
parser.add_argument('--data_train', type=str, default='DIV2K', help='train dataset name')
parser.add_argument('--data_test', type=str, default=['Set5'], help='validation/test dataset')
parser.add_argument('--n_train', type=int, default=800, help='number of training set')
parser.add_argument('--degrad_train', type=float, default={'type': 'B', 'min_sigma': 0.2, 'max_sigma': 2.6},
                    help='degradation settings for training, '
                         'type B for convolutive degradations, type N for additive degradations')
# parser.add_argument('--degrad_train', type=float, default={'type': 'N', 'min_sigma': 0, 'max_sigma': 30},
#                     help='degradation settings for training, '
#                          'type B for convolutive degradations, type N for additive degradations')
parser.add_argument('--degrad_test', type=float, default=[{'type': 'B', 'sigma': 0.2},
                                                        {'type': 'B', 'sigma': 1.3},
                                                        {'type': 'B', 'sigma': 2.6}],
                    help='degradation settings for testing/validation, '
                         'type B for convolutive degradations, type N for additive degradations')
# parser.add_argument('--degrad_test', type=float, default=[{'type': 'N', 'sigma': 0},
#                                                         {'type': 'N', 'sigma': 15},
#                                                         {'type': 'N', 'sigma': 30}],
#                     help='degradation settings for testing/validation, '
#                          'type B for convolutive degradations, type N for additive degradations')
parser.add_argument('--model_path', type=str, default='', help='path to save model')
parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--chop_forward', action='store_true', help='enable memory-efficient forward')

# Model specifications:
parser.add_argument('--n_homo_blocks', type=int, default=4, help='number of blocks in homo feature extraction module')
parser.add_argument('--n_transi_layers', type=int, default=8, help='number of layers in transitive learning module')
parser.add_argument('--n_channels', type=int, default=64, help='number of feature channels')
parser.add_argument('--act', default=nn.ReLU(inplace=True), help='activation function')
parser.add_argument('--num_samples', type=int, default=4, help='number of samples in random cropping module')
parser.add_argument('--size_samples', type=int, default=32, help='spatial size of samples in random cropping module')
parser.add_argument('--pretrained_model', type=str, default='ResNet50', help='pretrained model for training DoTNet')

# Training/Testing specifications
parser.add_argument('--train', type=str, default='Train', help='True for training, False for testing')
parser.add_argument('--iter_epoch', type=int, default=2000, help='iteration in each epoch')
parser.add_argument('--start_epoch_DoT', default=100, type=int, help='start epoch for training')
parser.add_argument('--start_epoch_SR', default=100, type=int, help='start epoch for training')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--best_epoch_SR', type=int, default=0, help='best epoch of the trained SR_Models for testing')
parser.add_argument('--best_epoch_DoT', type=int, default=0, help='best epoch of the trained DoTNets for testing')
parser.add_argument('--resume_SR', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--resume_DoT', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--run_DoT', default=True, help='train DoTNet?')
parser.add_argument('--run_SR', default=True, help='train SR_Model?')

# Optimization specifications
parser.add_argument('--lr_DoT', type=float, default=0.0002, help='initial learning rate for training DoTNet')
parser.add_argument('--lr_DoT_step_size', type=int, default=100, help='learning rate decay per N epochs')
parser.add_argument('--lr_DoT_gamma', type=int, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--lr_SR', type=float, default=0.0002, help='initial learning rate for training SR_Model')
parser.add_argument('--lr_SR_step_size', type=int, default=100, help='learning rate decay per N epochs')
parser.add_argument('--lr_SR_gamma', type=int, default=0.5, help='learning rate decay factor for step decay')

args = parser.parse_args()
