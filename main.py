import os, shutil, time, torch, imageio, csv
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from importlib import import_module
from PIL import Image
from torch.autograd import Variable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter
from src.flops_counter import get_model_complexity_info
import utils
from model import Generator
import data.common as common
from option import args
from data import data
import src.degradation as degradation
from tqdm import tqdm


def main():
    global opt, SR_Model, DoTNet
    opt = utils.print_args(args)

    if opt.n_GPUs == 1:
        torch.cuda.set_device(opt.GPU_ID)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    print('Random Seed: ', opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print('===> Building SR_Model')
    SR_Model = Generator(opt.n_colors, opt.n_channels, opt.n_homo_blocks, opt.n_transi_layers, opt.act, opt.scale)
    DoTNet = getattr(import_module('model'), 'DoTNet_' + opt.pretrained_model)()

    print('===> Calculating NumParams & FLOPs')
    input_SR_size = (3, 160, 90)
    SR_flops, SR_params = get_model_complexity_info(SR_Model, input_SR_size, input_plus=True, as_strings=False,
                                                    print_per_layer_stat=False)
    print('-------------SR SR_Model-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(SR_params * 1e-3, SR_flops * 1e-9, input_SR_size))

    input_DoT_size = (3, opt.size_samples, opt.size_samples)
    DoT_flops, DoT_params = get_model_complexity_info(DoTNet, input_DoT_size, as_strings=False,
                                                      print_per_layer_stat=False)
    print('---------------DoTNet-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(DoT_params * 1e-3, DoT_flops * 1e-9, input_DoT_size))

    if opt.train == 'Train':
        SR_Model = utils.load_checkpoint(opt.resume_SR, opt.n_GPUs, SR_Model)
        DoTNet = utils.load_checkpoint(opt.resume_DoT, opt.n_GPUs, DoTNet)

        print('===> Setting GPU')
        if opt.cuda:
            if opt.n_GPUs > 1:
                SR_Model = torch.nn.DataParallel(SR_Model).cuda()
                para_SR = filter(lambda x: x.requires_grad, SR_Model.module.parameters())
                DoTNet = torch.nn.DataParallel(DoTNet).cuda()
                para_DoT = filter(lambda x: x.requires_grad, DoTNet.module.parameters())
            else:
                SR_Model = SR_Model.cuda()
                para_SR = filter(lambda x: x.requires_grad, SR_Model.parameters())
                DoTNet = DoTNet.cuda()
                para_DoT = filter(lambda x: x.requires_grad, DoTNet.parameters())

        print('===> Setting Optimizer')
        optimizer_SR = optim.Adam(para_SR, lr=opt.lr_SR)
        optimizer_DoT = optim.Adam(para_DoT, lr=opt.lr_DoT)
        scheduler_SR = optim.lr_scheduler.StepLR(optimizer_SR, step_size=opt.lr_SR_step_size, gamma=opt.lr_SR_gamma)
        scheduler_DoT = optim.lr_scheduler.StepLR(optimizer_DoT, step_size=opt.lr_DoT_step_size, gamma=opt.lr_DoT_gamma)

        print('===> Loading Training Dataset')
        train_dataloader = data(opt).get_loader()

        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        for epoch in range(1, opt.n_epochs + 1):
            print('===> Training TLSR on DIV2K-train')
            train(train_dataloader, optimizer_SR, optimizer_DoT, SR_Model, DoTNet, epoch, writer)
            if opt.run_SR > 0:
                utils.save_checkpoint(SR_Model, epoch, opt.model_path + '/SR Model')
            if opt.run_DoT:
                utils.save_checkpoint(DoTNet, epoch, opt.model_path + '/DoTNet')

            for i in range(len(opt.data_test)):
                for k in opt.degrad_test:
                    valid_path = opt.dir_data + 'Test/' + opt.data_test[i]
                    PSNR, DoT_est = validation(valid_path, SR_Model, DoTNet, opt.scale, k['type'], k['sigma'],
                                               f_csv=None)
                    writer.add_scalar(str(opt.data_test[i]) + '-PSNR/TLSR_D{:.2f}'.format(k['sigma']), PSNR, epoch)
                    writer.add_scalar(str(opt.data_test[i]) + '-DoT_est/TLSR_D{:.2f}'.format(k['sigma']), DoT_est,
                                      epoch)
            torch.cuda.empty_cache()
            scheduler_SR.step()
            scheduler_DoT.step()
        writer.close()
    elif opt.train == 'Test':
        resume_SR = opt.model_path + '/SR Model/model_epoch_' + str(opt.best_epoch_SR) + '.pth'
        SR_Model = utils.load_checkpoint(resume_SR, opt.n_GPUs, SR_Model)
        resume_DoT = opt.model_path + '/DoTNet/model_epoch_' + str(opt.best_epoch_DoT) + '.pth'
        DoTNet = utils.load_checkpoint(resume_DoT, opt.n_GPUs, DoTNet)

        if opt.cuda:
            if opt.n_GPUs > 1:
                SR_Model = torch.nn.DataParallel(SR_Model).cuda()
                DoTNet = torch.nn.DataParallel(DoTNet).cuda()
            else:
                SR_Model = SR_Model.cuda()
                DoTNet = DoTNet.cuda()

        for i in range(len(opt.data_test)):
            for k in opt.degrad_test:
                with open(opt.model_path + '/' + opt.data_test[i] + '_' + k['type'] + str(k['sigma']) + '.csv',
                          'w', newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(['image_name', 'PSNR', 'SSIM', 'Time'])
                    valid_path = opt.dir_data + 'Test/' + opt.data_test[i]
                    validation(valid_path, SR_Model, DoTNet, opt.scale, k['type'], k['sigma'], f_csv)
        torch.cuda.empty_cache()
    else:
        raise InterruptedError


def train(training_dataloader, optimizer_SR, optimizer_DoT, SR_Model, DoTNet, epoch, writer):
    criterion_MAE = nn.L1Loss(reduction='mean').cuda()
    SR_Model.train()
    DoTNet.train()
    normalize_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().view(1, 3, 1, 1)
    normalize_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().view(1, 3, 1, 1)
    loss_DoT = 0
    loss_SR = 0
    with tqdm(total=len(training_dataloader)) as pbar:
        for iteration, HR_img in enumerate(training_dataloader):
            if HR_img.shape[0] == opt.batch_size:
                HR_img = Variable(HR_img, volatile=False)

                if opt.run_DoT:  # train TLSR model with transitive degradations
                    DoT_real = np.random.uniform(0, 1, size=opt.batch_size // 2)
                    DoT_real = np.append(DoT_real, np.zeros(opt.batch_size // 4))
                    DoT_real = np.append(DoT_real, np.ones(opt.batch_size // 4))
                elif opt.run_SR:  # train baseline models with extreme degradations
                    if np.random.random() < 0.5:
                        DoT_real = np.zeros(opt.batch_size)
                    else:
                        DoT_real = np.ones(opt.batch_size)

                sigma = DoT_real * (opt.degrad_train['max_sigma'] - opt.degrad_train['min_sigma']) + \
                        opt.degrad_train['min_sigma']
                if opt.degrad_train['type'] == 'B':
                    blur_size = 15
                    noise = False
                    noise_level = 0
                    blur_sigma = sigma
                elif opt.degrad_train['type'] == 'N':
                    blur_size = False
                    blur_sigma = 0
                    noise = True
                    noise_level = sigma / 255
                else:
                    raise InterruptedError

                # ----------------------Preparing degraded LR images------------------------
                prepro = degradation.SRMDPreprocessing(opt.scale, random=False,
                                                       kernel=blur_size, sig=blur_sigma,
                                                       noise=noise, noise_high=noise_level)
                if opt.cuda:
                    HR_img = HR_img.cuda()
                    DoT_real = torch.from_numpy(DoT_real).float().cuda()
                else:
                    DoT_real = torch.from_numpy(DoT_real).float()
                LR_img = prepro(HR_img)
                LR_img = (LR_img - normalize_mean) / normalize_std
                HR_img = (HR_img - normalize_mean) / normalize_std
                LR_img = Variable(LR_img)

                # ----------------------Updating DoTNet------------------------
                if opt.run_DoT:
                    for p in DoTNet.parameters():
                        p.requires_grad = True
                    for p in SR_Model.parameters():
                        p.requires_grad = False
                    DoT_real = DoT_real.repeat(opt.num_samples)
                    cropped_samples = utils.random_cropping(LR_img, opt.size_samples, opt.num_samples)
                    DoT_est = DoTNet(cropped_samples)
                    if not opt.run_SR:
                        optimizer_DoT.zero_grad()
                        loss_DoT = criterion_MAE(DoT_est, DoT_real)
                        loss_DoT.backward()
                        optimizer_DoT.step()

                    DoT_est = DoT_est.view(opt.num_samples, opt.batch_size)
                    DoT_est = torch.mean(DoT_est, dim=0)

                # ----------------------Updating SR Model------------------------
                if opt.run_SR:
                    for p in DoTNet.parameters():
                        p.requires_grad = False
                    for p in SR_Model.parameters():
                        p.requires_grad = True
                    optimizer_SR.zero_grad()
                    if opt.run_DoT:
                        LR_img_plus = {'value': LR_img, 'DoT': DoT_est.detach(), 'transi_learn': True}
                    else:
                        LR_img_plus = {'value': LR_img, 'DoT': DoT_real, 'transi_learn': False}

                    SR_img = SR_Model(LR_img_plus)
                    loss_SR = criterion_MAE(SR_img, HR_img)
                    loss_SR.backward()
                    optimizer_SR.step()

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(_epoch=epoch,
                                 _lr_DoT=optimizer_DoT.param_groups[0]['lr'],
                                 _lr_SR=optimizer_SR.param_groups[0]['lr'],
                                 loss_DoT='%.4f' % loss_DoT,
                                 loss_SR='%.4f' % loss_SR)

                if (iteration + 1) % 50 == 0:
                    niter = (epoch - 1) * len(training_dataloader) + iteration + 1
                    writer.add_scalar('Loss/loss_SR', loss_SR, niter)
                    writer.add_scalar('Loss/loss_DoT', loss_DoT, niter)


def validation(valid_path, SR_Model, DoTNet, scale, degrad_type, degrad_sigma, f_csv):
    SR_Model.eval()
    DoTNet.eval()
    count = 0
    Avg_DoT = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_Time = 0
    file = os.listdir(valid_path)
    file.sort()
    length = file.__len__()
    if degrad_type == 'B':
        blur_size = 15
        blur_sigma = degrad_sigma
        noise = False
        noise_level = 0
    elif degrad_type == 'N':
        blur_size = False
        blur_sigma = 0
        noise = True
        noise_level = degrad_sigma / 255
    else:
        raise InterruptedError

    normalize_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().view(1, 3, 1, 1)
    normalize_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().view(1, 3, 1, 1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        with tqdm(total=length) as pbar:
            for idx_img in range(length):
                torch.cuda.empty_cache()
                img_name = file[idx_img].split('.png')[0]
                HR_img = imageio.imread(valid_path + '/' + img_name + '.png')
                HR_img = common.set_channel(HR_img, opt.n_colors)
                HR_img = common.np2Tensor(HR_img, opt.rgb_range)
                HR_img = Variable(HR_img).view(1, HR_img.shape[0], HR_img.shape[1], HR_img.shape[2])
                if opt.cuda:
                    HR_img = HR_img.cuda()
                prepro = degradation.SRMDPreprocessing(scale, random=False, kernel=blur_size, noise=noise,
                                                       sig=[blur_sigma], sig_min=0, sig_max=0,
                                                       rate_iso=1.0, scaling=3, noise_high=[noise_level])
                LR_img = prepro(HR_img)
                LR_img = (LR_img - normalize_mean) / normalize_std
                start.record()

                cropped_samples = utils.random_cropping(LR_img, opt.size_samples, opt.num_samples)
                DoT = DoTNet(cropped_samples).mean()
                LR_img_plus = {'value': LR_img, 'num_samples': opt.num_samples, 'DoT': DoT, 'transi_learn': True}
                SR_img = SR_Model(LR_img_plus)
                end.record()
                torch.cuda.synchronize()
                SR_img = SR_img * normalize_std + normalize_mean
                SR_img = SR_img.data[0].cpu()

                Time = start.elapsed_time(end) * 1e-3
                PSNR = utils.calc_PSNR(SR_img, HR_img.data[0].cpu(), rgb_range=opt.rgb_range, shave=opt.scale)
                SSIM = utils.calc_SSIM(SR_img, HR_img.data[0].cpu(), rgb_range=opt.rgb_range, shave=opt.scale)
                DoT = DoT.data.cpu().numpy()

                if f_csv:
                    f_csv.writerow([img_name, PSNR, SSIM, Time, DoT])

                Avg_PSNR += PSNR
                Avg_SSIM += SSIM
                Avg_Time += Time
                Avg_DoT += DoT
                count = count + 1
                if opt.n_colors > 1:
                    SR_img = SR_img.mul(255).clamp(0, 255).round()
                    SR_img = SR_img.numpy().astype(np.uint8)
                    SR_img = SR_img.transpose((1, 2, 0))
                    SR_img = Image.fromarray(SR_img)
                else:
                    SR_img = SR_img.clamp(0, 255).round()
                    SR_img = SR_img[0, :, :].numpy().astype(np.float32)
                    SR_img = Image.fromarray(SR_img).convert('L')

                SR_path = opt.model_path + '/SRResults/' + valid_path.split('Test/')[1] + '/' + \
                          degrad_type + str(degrad_sigma)
                if not os.path.exists(SR_path):
                    os.makedirs(SR_path)
                SR_img.save(SR_path + '/' + img_name + '.png')

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(Degradation=degrad_type + str(degrad_sigma), DoT_Est='%.2f' % (Avg_DoT / count),
                                 PSNR='%.4f' % (Avg_PSNR / count), SSIM='%.4f' % (Avg_SSIM / count))
    if f_csv:
        f_csv.writerow(['Avg', Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count, Avg_DoT / count])
    return Avg_PSNR / count, Avg_DoT / count


if __name__ == '__main__':
    main()
