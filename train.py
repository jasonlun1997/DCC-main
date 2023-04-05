import torch
import torch.backends.cudnn as cudnn
import importlib
import math
import imageio
import scipy.io
import os
from torch import nn, optim
from torchvision import transforms, models
from tqdm import tqdm
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from utils.imresize import imresize
from collections import OrderedDict
from ignite.engine import create_supervised_trainer
from ignite.handlers import Checkpoint

'''bounding process'''
def mir(x, edge): 
    if (x < 0): 
        output = -x
    elif (x >= edge):  
        output = edge - (x - edge + 2)
    elif (x == 0):  
        output = 0
    else: 
        output = x
    return output

'''recognizer load model'''
def load_recog_model(model, checkpoint_position, device, to_save):
    checkpoint = torch.load(checkpoint_position, map_location=device) 
    Checkpoint.load_objects(to_load = to_save, checkpoint = checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()
    return model

'''single image Resize'''
def apply_test_transforms(inp):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.CenterCrop( int(432/args.scale_factor) )])
    out=transform(inp)
    return out

'''single image go into model'''
def predict(mmodel, im):
    if len(im.shape)==2:
        H, W = im.shape
        im3 = torch.stack([im,im,im])

    im_as_tensor = apply_test_transforms(im3)
    minibatch = torch.stack([im_as_tensor])
    if torch.cuda.is_available():
        minibatch = minibatch.cuda()

    pred = mmodel(minibatch)
    probs = F.softmax(pred,dim=1)
    a = probs.cpu().numpy()
    return a[0][0], a[0][1]

def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)

    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = './pth/' + args.model_name + '_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                        str(args.scale_factor) + 'x_model.pth'
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)

    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)

    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer)
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))

        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' Validation '''
        step = 1
        if (idx_epoch + 1)%step==0 or idx_epoch > args.epoch-step:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('cb_VAL_epoch_%02d' % (idx_epoch + 1)) 
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
                    excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

                    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

                    psnr_testset.append(psnr_epoch_test)
                    ssim_testset.append(ssim_epoch_test)
                    logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                    idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
                    pass
                psnr_mean_test = float(np.array(psnr_testset).mean())
                ssim_mean_test = float(np.array(ssim_testset).mean())
                logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
                                  % (psnr_mean_test, ssim_mean_test))
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        pass
    pass

def train(train_loader, device, net, criterion, optimizer):
    ''' training one epoch '''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)      # low resolution
        label = label.to(device)    # high resolution
        out = net(data, data_info)
        loss = criterion(out, label, data_info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train

def test(test_loader, device, net, net_re, save_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []                 
    fsim_iter_test = []
    downsample_scheme = ''
    angRes = args.angRes_in

    '''recognizer define'''
    model = models.swin_transformer.swin_b(weights='DEFAULT') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=0)    
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    n_inputs = model.head.in_features   
    last_layer = nn.Linear(n_inputs, 2) 
    model.head = last_layer             

    '''load lf data'''
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes  
        Hr_SAI_y = Hr_SAI_y
        Sr_SAI_cbcr = Sr_SAI_cbcr

        p0_list = []
        p1_list = []
        pp0_list = []
        pp1_list = []
        hei , wei = Lr_SAI_y.shape
        
        if (args.quality_enhancement_method == True):
            checkpoint_position = args.position_recongnizer        
            checkpoint_scheme = args.scheme_recongnizer
            
            '''recognize downsample position'''
            model_p = load_recog_model(model,checkpoint_position, device, to_save)
            model_p.eval()
            for u in range(Lr_angRes_in):
                for v in range(Lr_angRes_in):
                    single_SAI = Lr_SAI_y[ (hei//angRes)*u : (hei//angRes)*(u+1), (wei//angRes)*v : (wei//angRes)*(v+1)]
                    
                    prob_0, prob_1 =predict(model_p,single_SAI)
                    p0_list.append(prob_0)
                    p1_list.append(prob_1)

            '''recognize downsample scheme'''
            model_s = load_recog_model(model,checkpoint_scheme, device, to_save)
            model_s.eval()
            for u in range(Lr_angRes_in):
                for v in range(Lr_angRes_in):
                    single_SAI = Lr_SAI_y[ (hei//angRes)*u : (hei//angRes)*(u+1), (wei//angRes)*v : (wei//angRes)*(v+1)]
                    pprob_0, pprob_1 =predict(model_s,single_SAI)
                    pp0_list.append(pprob_0)
                    pp1_list.append(pprob_1)

            m0 = float(np.array(p0_list).mean())    #center
            m1 = float(np.array(p1_list).mean())    #non-center
            mm0 = float(np.array(pp0_list).mean())  #420A
            mm1  = float(np.array(pp1_list).mean()) #bicubic

        #if you do not want to see each LF probabilty, you can comment out 
            # print('The center is %.5f, non-center is %.5f, 420A is %.5f, bicubic is %.5f' % (m0,m1,mm0,mm1))    
            
            if (m0 >= m1):
                if (mm1 >= mm0):
                    downsample_scheme = 'bicubic'
                else :
                    downsample_scheme = '420A'
            else :
                downsample_scheme = '420D'

            '''re-interpolation'''
            if (downsample_scheme == '420D'):            
                for u in range(angRes):
                    for v in range(angRes):
                        _, _, He, Wi = Hr_SAI_y.shape
                        He = (He//angRes)
                        Wi = (Wi//angRes)
                        tmp_Lr_rgb = Lr_SAI_y[u*(hei//angRes): (u+1)*(hei//angRes), v*(wei//angRes): (v+1)*(wei//angRes)]
                        tmp_Lr_y = tmp_Lr_rgb[:, :]

                        '''bicubic Upsample'''
                        # tmp_Hr_y = cv2.resize(tmp_Lr_y, (Wi, He), interpolation=cv2.INTER_CUBIC)  #opencv bicubic (option)
                        tmp_Hr_y = imresize(tmp_Lr_y.cpu(), scalar_scale=args.scale_factor)         #project initial (option)

                        '''shifting & remap'''
                        map_x = np.zeros((tmp_Hr_y.shape[0], tmp_Hr_y.shape[1]), dtype=np.float32)
                        map_y = np.zeros((tmp_Hr_y.shape[0], tmp_Hr_y.shape[1]), dtype=np.float32)
                        for i in range(math.floor(He)):    
                            for j in range(math.floor(Wi)):
                                if (args.scale_factor == 2):
                                    map_x[i,j] = j + 0.5
                                    map_y[i,j] = i + 0.5
                                elif (args.scale_factor == 4):
                                    map_x[i,j] = j + 1.5
                                    map_y[i,j] = i + 1.5
                        tmp_Hr_y = cv2.remap(tmp_Hr_y, map_x, map_y, cv2.INTER_LINEAR) 

                        '''downsample'''
                        if (args.scale_factor == 4):
                            tmp_Lr_y = np.zeros((math.floor(He/4), math.floor(Wi/4)), dtype='single')                                
                            for i in range(math.floor(He/4)):
                                for j in range(math.floor(Wi/4)):
                                    tmp_Lr_y[i, j] = (tmp_Hr_y[i*4, j*4] + tmp_Hr_y[mir(i*4+1,He), j*4]
                                                    + tmp_Hr_y[i*4, mir(j*4+1,Wi)] + tmp_Hr_y[mir(i*4+1,He), mir(j*4+1,Wi)] 
                                                    + tmp_Hr_y[mir(i*4,He), mir(j*4+2,Wi)] + tmp_Hr_y[mir(i*4,He), mir(j*4+3,Wi)] 
                                                    + tmp_Hr_y[mir(i*4+1,He), mir(j*4+2,Wi)] + tmp_Hr_y[mir(i*4+1,He), mir(j*4+3,Wi)]
                                                    + tmp_Hr_y[mir(i*4+2,He), mir(j*4,Wi)] + tmp_Hr_y[mir(i*4+2,He), mir(j*4+1,Wi)]
                                                    + tmp_Hr_y[mir(i*4+3,He), mir(j*4,Wi)] + tmp_Hr_y[mir(i*4+3,He), mir(j*4+1,Wi)]
                                                    + tmp_Hr_y[mir(i*4+2,He), mir(j*4+2,Wi)] + tmp_Hr_y[mir(i*4+2,He), mir(j*4+3,Wi)]
                                                    + tmp_Hr_y[mir(i*4+3,He), mir(j*4+2,Wi)] + tmp_Hr_y[mir(i*4+3,He), mir(j*4+3,Wi)]) / 16 # 420A
                        elif (args.scale_factor == 2):
                            tmp_Lr_y = np.zeros((math.floor(He/2), math.floor(Wi/2)), dtype='single')                                
                            for i in range(math.floor(He/2)):    
                                for j in range(math.floor(Wi/2)):
                                    tmp_Lr_y[i, j] = (tmp_Hr_y[i*2, j*2] + tmp_Hr_y[mir(i*2+1,He), j*2]
                                                    + tmp_Hr_y[i*2, mir(j*2+1,Wi)] + tmp_Hr_y[mir(i*2+1,He), mir(j*2+1,Wi)]) / 4 # 420A
                    
                        Lr_SAI_y[u*(hei//angRes) : (u+1)*(hei//angRes), v*(wei//angRes) : (v+1)*(wei//angRes)] = torch.from_numpy(tmp_Lr_y)
        #end of if (args.quality_enhancement_method == True):

        ''' bicubic-Crop LFs into Patches '''
        subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                               args.angRes_in * args.patch_size_for_test * args.scale_factor)

        ''' SR the Patches '''
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                if (downsample_scheme == '420A'):      
                    net_re.eval()
                    torch.cuda.empty_cache()
                    out = net_re(tmp.to(device), data_info)
                else:
                    net.eval()
                    torch.cuda.empty_cache()
                    out = net(tmp.to(device), data_info)
                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
        subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

        ''' Restore the Patches to LFs '''
        Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
        Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')#Sr_SAI_y

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim, fsim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        fsim_iter_test.append(fsim)
        LF_iter_test.append(LF_name[0])

        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')   
            views_dir.mkdir(exist_ok=True)            

            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            # save the SAI
            path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
            imageio.imwrite(path, Sr_SAI_rgb) 
            # save the center view
            img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
            path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
            imageio.imwrite(path, img) 
            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'  
                    imageio.imwrite(path, img)   
                    pass
                pass

            save_path = str(save_dir_)
            if not (os.path.exists(save_path)):
                os.makedirs(save_path)
            scipy.io.savemat(save_path + '/' + LF_name[0] + '.mat',   
                            {'LF': Sr_SAI_y.cpu().numpy()})
            pass

        pass

    return psnr_iter_test, ssim_iter_test, fsim_iter_test, LF_iter_test

if __name__ == '__main__':
    from option import args

    main(args)
