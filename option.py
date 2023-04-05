import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--angRes', type=int, default=5, help="angular resolution")
parser.add_argument('--scale_factor', type=int, default=4, help="4, 2")
parser.add_argument('--model_name', type=str, default='DistgSSR', help="model name")    # DistgSSR # LF_IINet # DFnet # LF_InterNet 
parser.add_argument('--use_pre_ckpt', type=bool, default=True, help="use pre model ckpt")   #False#True

parser.add_argument('--path_pre_pth', type=str, default='./pth/SR/bicubic/DistgSSR_4xSR_5x5.pth.tar', help="path for pre model ckpt")   
parser.add_argument('--data_name', type=str, default='ALL', help='HCI, ALL(of Five Datasets)')  #EPFL, HCI, HCI_old, INRIA_Lytro, Stanford_Gantry
parser.add_argument('--path_for_train', type=str, default='../BasicLFSR-main/data_for_training/SR_5x5_2x/')
parser.add_argument('--path_for_test', type=str, default='../BasicLFSR-main/data_for_test/SR_420D_5x5_4x/')

parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--batch_size', type=int, default=8) #DistgSSR 8    #IInet 10   #Inter 12 
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')# DistgSSR.IInet 2e-4 # Inter 5e-4
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')# DistgSSR 15 # IInet 10 #Inter 10
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')   #every step decrease lr by gamma factor
parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')# DistgSSR.IInet 50 # Inter 40
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')#2
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

# for test.py
parser.add_argument('--path_re_pth', type=str, default='./pth/SR/420A/DistgSSR_5x5_4x_420A_model.pth', help="path for pre model ckpt") 
parser.add_argument('--quality_enhancement_method', type=bool, default=True, help="use DCC-based method")

parser.add_argument('--position_recongnizer', type=str, default='./pth/recognizer/position_recognizer_4x.pt', help="path for position recognizer")
parser.add_argument('--scheme_recongnizer', type=str, default='./pth/recognizer/scheme_recognizer_4x.pt', help="path for scheme recognizer")
parser.add_argument('--result_dir', type=str, default='420D_our', help="name for result dir ")

args = parser.parse_args()
args.angRes_in = args.angRes
args.angRes_out = args.angRes
args.patch_size_for_test = 32
args.stride_for_test = 16
args.minibatch_for_test = 16

del args.angRes