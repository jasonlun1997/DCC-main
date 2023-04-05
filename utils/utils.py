import numpy as np
import torch
import logging
import xlwt
import torch.nn.functional as F
import phasepack.phasecong as pc
import cv2
from skimage import metrics
from pathlib import Path
from option import args
from einops import rearrange

class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.write(0, 4, 'FSIM')#
        
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10        
        self.worksheet.col(4).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test, fsim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i], fsim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        fsim_epoch_test = float(np.array(fsim_iter_test).mean())
        
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test, fsim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test, fsim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)        
        self.worksheet.write(self.sum, 4, '%.6f' % fsim_iter_test)#
        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(args): # save model dir
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
    
    log_dir = log_dir.joinpath(path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    return log_dir, checkpoints_dir, results_dir


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, str):
        if args.local_rank <= 0:
            self.logger.info(str)
            print(str)

def cal_metrics(args, label, out,):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    FSIM = np.zeros(shape=(B, U, V), dtype='float32')
    
    for b in range(B):# batch
        for u in range(U):# spatial resolution row
            for v in range(V):# spatial resolution column
                # FSIM[b, u, v] = fsim(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())    # fsim   # triger use or not
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())
                SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),out_y[b, u, :, v, :].numpy(),gaussian_weights=True, )
                pass

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)
    FSIM_mean = FSIM.sum() #/ np.sum(FSIM > 0)

    return PSNR_mean, SSIM_mean, FSIM_mean


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y

'''ref:https://github.com/up42/image-similarity-measures/blob/1bddf8c30337f7ec7b0181126e76a3d26efcf360/image_similarity_measures/quality_metrics.py#L137'''
def fsim(imageRef, imageDis):
    T1 = 0.85
    T2 = 160
    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []

    # Calculate the PC for original and predicted images
    pc1_2dim = pc(
        imageRef[:, :], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978   #[:, :, i]
    )
    pc2_2dim = pc(
        imageDis[:, :], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978   #[:, :, i]
    )

    # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
    # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
    # calculate the sum of all these 6 arrays.
    pc1_2dim_sum = np.zeros((imageRef.shape[0], imageRef.shape[1]), dtype=np.float64)
    pc2_2dim_sum = np.zeros(
        (imageDis.shape[0], imageDis.shape[1]), dtype=np.float64
    )
    for orientation in range(6):
        pc1_2dim_sum += pc1_2dim[4][orientation]
        pc2_2dim_sum += pc2_2dim[4][orientation]

    # Calculate GM for original and predicted images based on Scharr operator
    scharrx = cv2.Scharr(imageRef[:, :], cv2.CV_16U, 1, 0)   
    scharry = cv2.Scharr(imageRef[:, :], cv2.CV_16U, 0, 1)  
    gm1 = np.sqrt(scharrx**2 + scharry**2)

    scharrx_2 = cv2.Scharr(imageDis[:, :], cv2.CV_16U, 1, 0) 
    scharry_2 = cv2.Scharr(imageDis[:, :], cv2.CV_16U, 0, 1) 
    gm2 = np.sqrt(scharrx_2**2 + scharry_2**2)  

    # Calculate similarity measure for PC1 and PC2
    numerator = 2 * pc1_2dim_sum * pc2_2dim_sum + T1
    denominator = pc1_2dim_sum**2 + pc2_2dim_sum**2 + T1
    S_pc = numerator / denominator

    # Calculate similarity measure for GM1 and GM2
    numerator_2 = 2 * gm1 * gm2 + T2
    denominator_2 = gm1**2 + gm2**2 + T2
    S_g = numerator_2 / denominator_2

    S_l = (S_pc**alpha) * (S_g**beta)

    numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)
