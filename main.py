import os, shutil, time, torch, imageio, csv, random

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from importlib import import_module
from PIL import Image
from torch.autograd import Variable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter
import utils
import model as architecture
import data.common as common
from option import args
from data import data
import src.degradation as degradation
from tqdm import tqdm
import torch.nn.functional as F
# from pytorch_msssim import SSIM
from src.cal_complexity import profile_origin
import warnings

warnings.filterwarnings("ignore")


def main():
    global opt, normalize_mean, normalize_std
    opt = utils.print_args(args)

    normalize_mean = torch.from_numpy(np.array([0.466, 0.448, 0.403])).float().view(1, 3, 1, 1)
    normalize_std = torch.from_numpy(np.array([0.242, 0.234, 0.246])).float().view(1, 3, 1, 1)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        # os.environ['CUDA_VISIBLE_DEVICES'] = opt.GPU_ID
        normalize_mean, normalize_std = normalize_mean.cuda(), normalize_std.cuda()

    cudnn.benchmark = True

    print('===> Building SR_Model')
    print("===> Building model")
    model = {
        'SR': architecture.Generator(opt.n_colors,
                                     opt.n_channels,
                                     opt.n_homo_blocks,
                                     opt.n_transi_layers,
                                     opt.n_homo_width,
                                     opt.n_transi_width,
                                     opt.act,
                                     opt.scale),
        'DoTNet': architecture.DoTNet(opt.n_colors, opt.size_samples)
    }

    optimizer = {
        'SR': None,
        'DoTNet': None
    }

    scheduler = {
        'SR': None,
        'DoTNet': None
    }

    print('===> Calculating NumParams & FLOPs')
    input = {"value": torch.FloatTensor(1, opt.n_colors, 480 // opt.scale, 360 // opt.scale),
             "num_samples": 16,
             "DoT": torch.zeros(1),
             "transi_learn": True}
    SR_macs, SR_params = profile_origin(model['SR'], inputs=(input,), verbose=False)
    print('-------------SR Model-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(SR_params * 1e-3, SR_macs * 1e-9, input['value'].shape))

    input = torch.FloatTensor(opt.num_samples, opt.n_colors, opt.size_samples, opt.size_samples)
    DoT_macs, DoT_params = profile_origin(model['DoTNet'], inputs=(input,), verbose=False)
    print('-------------SR Model-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(DoT_params * 1e-3, DoT_macs * 1e-9, input.shape))

    if opt.train == 'Train':
        model['SR'] = utils.load_checkpoint(opt.resume_SR, opt.n_GPUs, model['SR'], is_cuda=opt.cuda)
        model['DoTNet'] = utils.load_checkpoint(opt.resume_DoT, opt.n_GPUs, model['DoTNet'], is_cuda=opt.cuda)

        print("===> Setting GPU")
        for item in model:
            if opt.n_GPUs > 1 and opt.cuda:
                model[item] = torch.nn.DataParallel(model[item]).cuda()
                para = filter(lambda x: x.requires_grad, model[item].module.parameters())
            else:
                model[item] = model[item].cuda() if opt.cuda else model[item]
                para = filter(lambda x: x.requires_grad, model[item].parameters())
            optimizer[item] = optim.Adam(params=para, lr=opt.lr[item])
            scheduler[item] = optim.lr_scheduler.StepLR(optimizer[item],
                                                        step_size=opt.lr_step_size[item],
                                                        gamma=opt.lr_gamma[item])
            model[item].train()

        print('===> Loading Training Dataset')
        train_dataloader = data(opt).get_loader()

        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        writer.add_scalar('Complexity/SR_Params(K)', SR_params * 1e-3, 0)
        writer.add_scalar('Complexity/SR_FLOPs(G)', SR_macs * 1e-9, 0)
        writer.add_scalar('Complexity/DoT_Params(K)', DoT_params * 1e-3, 0)
        writer.add_scalar('Complexity/DoT_FLOPs(G)', DoT_macs * 1e-9, 0)

        for epoch in range(1, opt.n_epochs + 1):
            print('===> Training TLSR on DIV2K-train')
            train(train_dataloader, optimizer, model, epoch, writer)
            if opt.run_SR > 0:
                utils.save_checkpoint(model['SR'], epoch, opt.model_path + '/SR Model')
            if opt.run_DoT and opt.DoT == 'est':
                utils.save_checkpoint(model['DoTNet'], epoch, opt.model_path + '/DoTNet')

            print('===> Testing TLSR on benchmarks')
            with torch.no_grad():
                for i in range(len(opt.data_test)):
                    for k in opt.degrad_test:
                        item_PSNR = {}
                        item_TIME = {}
                        item_DoT = {}
                        valid_path = opt.dir_data + 'Test/' + opt.data_test[i] + '/HR'
                        PSNR, Time, DoT_gt, _ = validation(valid_path, model, opt.scale, k['type'], [k['sigma']], f_csv=None, transi_learn='est')
                        item_PSNR['TLSR_GT'] = PSNR
                        item_TIME['TLSR_GT'] = Time
                        item_DoT['GT'] = DoT_gt
                        writer.add_scalars(str(opt.data_test[i]) + '/' + k['type'] + str(k['sigma']) + '-PSNR', item_PSNR, epoch)
                        writer.add_scalars(str(opt.data_test[i]) + '/' + k['type'] + str(k['sigma']) + '-TIME', item_TIME, epoch)
                        writer.add_scalars(str(opt.data_test[i]) + '/' + k['type'] + str(k['sigma']) + '-DoT', item_DoT, epoch)

            # torch.cuda.empty_cache()
            scheduler['SR'].step()
            scheduler['DoTNet'].step()
        writer.close()
    elif opt.train == 'Test':
        opt.threads = 1
        model['SR'] = utils.load_checkpoint(opt.resume_SR, opt.n_GPUs, model['SR'], is_cuda=opt.cuda)
        model['DoTNet'] = utils.load_checkpoint(opt.resume_DoT, opt.n_GPUs, model['DoTNet'], is_cuda=opt.cuda)

        print("===> Setting GPU")
        for item in model:
            model[item] = model[item].cuda() if opt.cuda else model[item]
            model[item].eval()

        SR_path = opt.model_path + '/SRResults/'
        if not os.path.exists(SR_path):
            os.makedirs(SR_path)
        with torch.no_grad():
            for i in range(len(opt.data_test)):
                for k in opt.degrad_test:
                    valid_path = opt.dir_data + 'Test/' + opt.data_test[i]
                    with open(SR_path + '/' + opt.data_test[i] + '_' + k['type'] + str(k['sigma']) + '_TLSR.csv',
                              'w', newline='') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(['image_name', 'PSNR', 'SSIM', 'Time', 'DoT'])
                        validation(valid_path, model, opt.scale, k['type'], [k['sigma']], f_csv=f_csv, transi_learn='est')
    else:
        raise InterruptedError


def train(training_dataloader, optimizer, model, epoch, writer):
    criterion_MAE = nn.L1Loss(reduction='mean').cuda()
    for item in model:
        model[item].train()
    with tqdm(total=len(training_dataloader), ncols=140) as pbar:
        for iteration, HR_img in enumerate(training_dataloader):
            HR_img = Variable(HR_img, requires_grad=False)
            if opt.cuda:
                HR_img = HR_img.cuda()
            # ----------------------Preparing degradation parameters------------------------
            if opt.run_DoT:  # train TLSR model with transitive degradations
                DoT_real = np.random.uniform(0, 1, size=3*opt.batch_size // 4)
                DoT_real = np.append(DoT_real, np.zeros(opt.batch_size // 8))
                DoT_real = np.append(DoT_real, np.ones(opt.batch_size // 8))

            elif opt.run_SR:  # train baseline models with primary degradations
                if random.random() < 0.5:
                    DoT_real = np.zeros(opt.batch_size)
                else:
                    DoT_real = np.ones(opt.batch_size)

            sigma = DoT_real * (opt.degrad_train['max_sigma'] - opt.degrad_train['min_sigma']) + \
                    opt.degrad_train['min_sigma']

            # ----------------------Preparing degraded LR images------------------------
            if opt.degrad_train['type'] != 'JPEG':
                if opt.degrad_train['type'] == 'B':
                    blur_size = opt.blur_size
                    noise = False
                    noise_level = 0
                    blur_sigma = sigma
                    angle = np.zeros_like(sigma)
                    aniso = False
                elif opt.degrad_train['type'] == 'B_aniso':
                    blur_size = opt.blur_size
                    noise = False
                    noise_level = 0
                    blur_sigma = 1.3 * np.ones_like(sigma)
                    angle = sigma
                    aniso = True
                elif opt.degrad_train['type'] == 'N':
                    blur_size = False
                    blur_sigma = 0
                    angle = np.zeros_like(sigma)
                    aniso = False
                    noise = True
                    noise_level = np.array(sigma) / 255
                else:
                    raise InterruptedError
                prepro = degradation.SRMDPreprocessing(opt.scale, random=False,
                                                       kernel=blur_size, sig=blur_sigma,
                                                       angle=angle, aniso=aniso, scaling=2.5,
                                                       noise=noise, noise_high=noise_level)
                LR_img = prepro(HR_img)
            else:

                LR_img = HR_img.clone()
                for i in range(opt.batch_size):
                    img = LR_img[i].mul(255).clamp(0, 255).round().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                    img = common.JPEG_compression(img, quality=sigma[i], n_channel=opt.n_colors)
                    img = common.set_channel(img, n_channel=opt.n_colors)
                    LR_img[i] = common.np2Tensor(img, opt.value_range).type_as(HR_img)

            if opt.cuda:
                DoT_real = torch.from_numpy(DoT_real).float().cuda()
            else:
                DoT_real = torch.from_numpy(DoT_real).float()
            LR_img = (LR_img - normalize_mean) / normalize_std
            HR_img = (HR_img - normalize_mean) / normalize_std
            LR_img = Variable(LR_img, requires_grad=True)

            # ----------------------Updating DoTNet------------------------
            if opt.run_DoT and opt.DoT == 'est':
                for p in model['DoTNet'].parameters():
                    p.requires_grad = True
                for p in model['SR'].parameters():
                    p.requires_grad = False
                DoT_real = DoT_real.repeat(opt.num_samples)
                cropped_samples = utils.random_cropping(LR_img, opt.size_samples, opt.num_samples)
                DoT_est = model['DoTNet'](cropped_samples)
                if (not opt.run_SR) or (iteration % 2 == 0):
                    optimizer['DoTNet'].zero_grad()
                    loss_DoT = criterion_MAE(DoT_est, DoT_real)
                    loss_DoT.backward()
                    optimizer['DoTNet'].step()
                DoT_est = DoT_est.contiguous().view(opt.num_samples, opt.batch_size).median(dim=0)[0].clamp(0, 1).detach()

            # ----------------------Updating SR Model------------------------
            if opt.run_SR:
                for p in model['DoTNet'].parameters():
                    p.requires_grad = False
                for p in model['SR'].parameters():
                    p.requires_grad = True
                optimizer['SR'].zero_grad()
                if opt.run_DoT:
                    if opt.DoT == 'est':
                        LR_img_plus = {'value': LR_img, 'DoT': DoT_est, 'transi_learn': True}
                    elif opt.DoT == 'gt':
                        LR_img_plus = {'value': LR_img, 'DoT': DoT_real, 'transi_learn': True}
                    else:
                        raise InterruptedError
                else:
                    LR_img_plus = {'value': LR_img, 'DoT': DoT_real, 'transi_learn': False}
                SR_img = model['SR'](LR_img_plus)
                loss_SR = criterion_MAE(SR_img, HR_img)
                loss_SR.backward()
                optimizer['SR'].step()

            time.sleep(0.01)
            pbar.update(1)
            pbar.set_postfix(_epoch=epoch,
                             _lr_DoT=optimizer['DoTNet'].param_groups[0]['lr'],
                             _lr_SR=optimizer['SR'].param_groups[0]['lr'],
                             loss_DoT='{:.3f}'.format(loss_DoT if 'loss_DoT' in locals().keys() else 0),
                             loss_SR='{:.3f}'.format(loss_SR if 'loss_SR' in locals().keys() else 0))

            if (iteration + 1) % 50 == 0:
                niter = (epoch - 1) * len(training_dataloader) + iteration + 1
                if 'loss_DoT' in locals().keys():
                    writer.add_scalar('Loss/loss_DoT', loss_DoT, niter)
                if 'loss_SR' in locals().keys():
                    writer.add_scalar('Loss/loss_SR', loss_SR, niter)
                if 'DoT_est_expand' in locals().keys():
                    item = {}
                    diff = torch.abs(DoT_real.view(opt.num_samples, opt.batch_size).mean(dim=0) - DoT_est)
                    for i in range(opt.batch_size):
                        item['batch%2d' % i] = diff.data[i]
                    writer.add_scalars('KerScore_Diff', item, niter)

def validation(valid_path, model, scale, degrad_type, degrad_sigma, f_csv, transi_learn='est', DoT=None):
    for item in model:
        model[item].eval()
    count = 0
    Avg_DoT = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_Time = 0
    file = os.listdir(valid_path)
    file.sort()
    length = file.__len__()

    if degrad_type != 'JPEG':
        if degrad_type == 'B':
            blur_size = opt.blur_size
            noise = False
            noise_level = 0
            blur_sigma = degrad_sigma
            angle = np.zeros_like(degrad_sigma)
            aniso = False
        elif degrad_type == 'B_aniso':
            blur_size = opt.blur_size
            noise = False
            noise_level = 0
            blur_sigma = 1.3 * np.ones_like(degrad_sigma)
            angle = degrad_sigma
            aniso = True
        elif degrad_type == 'N':
            blur_size = False
            blur_sigma = 0
            angle = np.zeros_like(degrad_sigma)
            aniso = False
            noise = True
            noise_level = np.array(degrad_sigma) / 255
        else:
            raise InterruptedError

        prepro = degradation.SRMDPreprocessing(scale, random=False,
                                               kernel=blur_size, sig=blur_sigma,
                                               angle=angle, aniso=aniso, scaling=2.5,
                                               noise=noise, noise_high=noise_level)

    if opt.cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        Time = 0
    if not transi_learn:
        if DoT != None:
            DoT = DoT*torch.ones(1)
        else:
            DoT = torch.ones(1) if degrad_sigma[0] > (opt.degrad_train['min_sigma'] + opt.degrad_train['max_sigma']) / 2 else torch.zeros(1)
        name = '_BL-D{:.1f}'.format(DoT.data[0].numpy()) # BaseLine
        name_method = 'Baseline'
    else:
        DoT = (np.array(degrad_sigma) - opt.degrad_train['min_sigma']) / (opt.degrad_train['max_sigma'] - opt.degrad_train['min_sigma'])
        DoT = torch.from_numpy(DoT).float()
        name = '_TL-D{:.1f}'.format(DoT.data[0].numpy())
        name_method = 'TLSR'
        if transi_learn == 'gt':
            name += '-GT'
            name_method = 'TLSR-GT'

    DoT_item = {}
    with torch.no_grad():
        with tqdm(total=length, ncols=140) as pbar:
            for idx_img in range(length):
                torch.cuda.empty_cache()
                img_name, ext = os.path.splitext(file[idx_img])
                HR_img = imageio.imread(valid_path + '/' + img_name + ext)

                if degrad_type != 'JPEG':
                    HR_img = common.set_channel(HR_img, opt.n_colors)
                    HR_img = common.np2Tensor(HR_img, opt.value_range)
                    HR_img = Variable(HR_img).view(1, HR_img.shape[0], HR_img.shape[1], HR_img.shape[2])
                    if opt.cuda:
                        HR_img = HR_img.cuda()
                    LR_img = prepro(HR_img)
                else:
                    HR_img = common.set_channel(HR_img, opt.n_colors)
                    LR_img = common.JPEG_compression(HR_img, degrad_sigma[0], opt.n_colors)
                    LR_img = common.set_channel(LR_img, opt.n_colors)
                    LR_img = common.np2Tensor(LR_img, opt.value_range)
                    LR_img = Variable(LR_img).view(1, LR_img.shape[0], LR_img.shape[1], LR_img.shape[2])
                    HR_img = common.np2Tensor(HR_img, opt.value_range)
                    HR_img = Variable(HR_img).view(1, HR_img.shape[0], HR_img.shape[1], HR_img.shape[2])
                    if opt.cuda:
                        HR_img = HR_img.cuda()
                        LR_img = LR_img.cuda()

                LR_img = (LR_img - normalize_mean) / normalize_std
                start.record()
                if transi_learn == 'est':
                    cropped_samples = utils.random_cropping(LR_img, opt.size_samples, opt.num_samples)
                    DoT_curr = model['DoTNet'](cropped_samples).median().unsqueeze(0).clamp(0, 1)
                else:
                    DoT_curr = DoT
                SR_img = model['SR'](
                    {'value': LR_img,
                     'num_samples': opt.num_samples,
                     'DoT': DoT_curr.type_as(LR_img),
                     'transi_learn': transi_learn}
                )
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end)
                SR_img = SR_img * normalize_std + normalize_mean
                SR_img = SR_img.data[0].cpu()
                DoT_data = DoT_curr.data[0].cpu().numpy()

                PSNR = utils.calc_PSNR(SR_img, HR_img.data[0].cpu(), opt.value_range, shave=opt.scale)
                SSIM = utils.calc_SSIM(SR_img, HR_img.data[0].cpu(), opt.value_range, shave=opt.scale)

                if f_csv:
                    f_csv.writerow([img_name, PSNR, SSIM, Time, DoT_data])

                Avg_PSNR += PSNR
                Avg_SSIM += SSIM
                Avg_Time += Time
                Avg_DoT += DoT_data

                count = count + 1
                if opt.n_colors > 1:
                    SR_img = SR_img.mul(255).clamp(0, 255).round()
                    SR_img = SR_img.numpy().astype(np.uint8)
                    SR_img = SR_img.transpose((1, 2, 0))
                    SR_img = Image.fromarray(SR_img)
                else:
                    SR_img = SR_img[0, :, :].mul(opt.value_range).clamp(0, opt.value_range).round().numpy().astype(np.uint8)
                    SR_img = Image.fromarray(SR_img).convert('L')

                SR_path = opt.model_path + '/SRResults/' + valid_path.split('Test/')[1] + '/' + \
                          degrad_type + str(degrad_sigma[0])
                if not os.path.exists(SR_path):
                    os.makedirs(SR_path)
                SR_img.save(SR_path + '/' + img_name + name + '.png')

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(Deg='X{:d}+'.format(opt.scale) + degrad_type + str(degrad_sigma[0]),
                                 METHOD=name_method,
                                 PSNR='%.3f' % (Avg_PSNR / count),
                                 SSIM='%.4f' % (Avg_SSIM / count),
                                 TAU='%.3f' % (Avg_DoT / count),
                                 TIME='{:.1f}ms'.format(Avg_Time / count))
    if f_csv:
        f_csv.writerow(['Avg', Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count, Avg_DoT / count])

    return Avg_PSNR / count, Avg_Time / count, Avg_DoT / count, DoT_item

if __name__ == '__main__':
    main()
