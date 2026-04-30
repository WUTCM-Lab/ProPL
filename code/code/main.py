import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from data.build_dataset import build_dataset
from models.build_model import build_model
from utils.evaluate import evaluate
from opt import args
from utils.loss import BceDiceLoss, BCELoss,DiceLoss
from monai.losses import DiceCELoss
import math
import warnings
import torchvision
import pandas as pd
import shutil
import time
warnings.filterwarnings("ignore", category=UserWarning)




def save_checkpoint(
        state,
        is_best,
        epoch,
        args = None,
        filename='latest.pth'):
    
    checkdir = args.root + "/semi/checkpoint/" +'expID'+str(args.expID) +'/'+ args.ckpt_name # + "/best.pth"
    if not os.path.exists(checkdir):
        os.makedirs(checkdir)
    file = filename.format(epoch)
    file_dir = os.path.join(checkdir, file)
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir, exist_ok=True)

    # file = '_'.join([file_dir, filename.format(epoch)])
    # pdb.set_trace()
    torch.save(state, file_dir)
    if is_best:
        # best_name = os.path.join  (
        #     file_dir,
        #     'model_best_' + name + '.pth')
        best_name = os.path.join(os.path.dirname(file_dir),'best.pth')
        shutil.copyfile(file_dir, best_name)

class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
 
    def forward(self, logits, targets):
        num = torch.sum(logits * targets, dim=(1, 2, 3))
        den = torch.sum((1 - logits) * targets + logits * (1 - targets), dim=(1, 2, 3))
        Tversky = (num / (den + self.eps)) * (1 - self.eps)
        return 1 - (Tversky * self.alpha + (1 - Tversky) * self.beta).mean()
 

def DeepSupSeg(pred, gt):
    d0 = pred  ##
    criterion = BceDiceLoss()
    
    loss0 = criterion(d0, gt)
    return loss0

def DeepSupSeg2(pred, gt):
    d0 = pred  ##
    criterion = DiceCELoss()
    loss0 = criterion(d0, gt)
    return loss0
def DeepSupInp(pred, gt, mask):
    criterion = nn.L1Loss()
    # loss = 0
    # for i in range(len(pred)):
    select_pred = torch.masked_select(pred[0], mask[0]>0.5)
    select_target = torch.masked_select(gt, mask[0]>0.5)
    loss = criterion(select_pred, select_target)
    # gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    return loss


def SupInp(pred, gt, mask):
    criterion = nn.L1Loss()
    select_pred = torch.masked_select(pred, mask>0.5)
    select_target = torch.masked_select(gt, mask>0.5)
    loss = criterion(select_pred, select_target)
    return loss


def EntropyLoss(pred, gt):
    gt = (gt > 0.5).float()
    criterion = BceDiceLoss()
    mse = criterion(pred, gt)
    entrop = -2 * torch.square(pred) * torch.log2(pred + 0.0001)
    return (mse + 0 * entrop).mean()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1-float(iter)/max_iter)**power)


def adjust_lr_rate(argsimizer, iter, total_batch):
    lr = lr_poly(args.lr, iter, args.nEpoch*total_batch, args.power)
    argsimizer.param_groups[0]['lr'] = lr
    return lr

def train():
    """load data"""
    train_l_data, train_u_data, valid_data = build_dataset(args)
    # train_l_data, train_u_data, valid_data, test_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    #train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # test_total_batch = int(len(test_data) / 1)
    """load model"""
    model = build_model(args)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)
    # optim = torch.optim.AdamW(model.parameters(),lr=args.lr)
    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    for epoch in range(args.nEpoch):
        model.train()
      
        print("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch)
        for batch_id, data_l in bar:
            #data_l, data_u = next(loader)
            
            #total_batch = len(train_u_dataloader)
            #total_batch = len(train_l_dataloader)
            itr = total_batch * epoch + batch_id

            img_l, gt = data_l
            if args.GPUs:
                img_l = [img_l[0].cuda(),{key:value.cuda() for key,value in img_l[1].items()}]
                gt = gt.cuda()[:,0:1,...]
            optim.zero_grad()
            mask_l, *_ = model(img_l)
            loss_l_seg = DeepSupSeg(mask_l.float(), gt.float()) 
            loss_l =  loss_l_seg
            loss = loss_l
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice,case_name = evaluate(model, valid_dataloader, val_total_batch,None)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" +'expID'+str(args.expID) +'/'+ args.ckpt_name + "/best.pth")


def train_semi():
    """load data"""
    train_l_data, train_u_data, valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)
    """load model"""
    model = build_model(args)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)
    # optim = torch.optim.AdamW(model.parameters(),lr=args.lr)
    
    # train
    print('\n---------------------------------')
    print('Start training_semi')
    print('---------------------------------\n')
    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    epoch_resume = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        epoch_resume = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        model_dict = model.state_dict()
        best = best_metric
        F1_best = best_metric
        print(len(model_dict))
        print(len(checkpoint['state_dict']))
        new_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print('upload parameter already')
        optim.load_state_dict(checkpoint['optimizer'])
        print(("=> loaded checkpoint '{}' (epoch {})"
               .format(args.resume, checkpoint['epoch'])))
        print(best_metric)
    
    old_time = time.time()
    for epoch in range(epoch_resume,args.nEpoch):
        # print("Epoch: {}, time:{}".format(epoch, time.time()-old_time))
        # old_time = time.time()
        model.train()
        print("Epoch: {}".format(epoch))
        loader = iter(zip(cycle(train_l_dataloader), train_u_dataloader))
        bar = tqdm(range(len(train_u_dataloader)))
        for batch_id in bar:
            data_l, data_u = next(loader)
            total_batch = len(train_u_dataloader)
            itr = total_batch * epoch + batch_id
            img_l, gt = data_l
            img_u = data_u
            
            if args.GPUs:
                img_l = [img_l[0].cuda(),{key:value.cuda() for key,value in img_l[1].items()}]
                gt = gt.cuda()[:,0:1,...]
                img_u = [img_u[0].cuda(),{key:value.cuda() for key,value in img_u[1].items()}]
            optim.zero_grad()
            #import pdb;pdb.set_trace()
            pred_l = model(img_l)
            mask,*_ = pred_l
            loss_l_seg = DeepSupSeg(mask, gt.float())
            # shape_l = F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False)
            # loss_l_shape = netD(shape_l)
            loss_l = loss_l_seg 
            print("sup loss:",loss_l)
            # import pdb;pdb.set_trace()
            predictions_mean = torch.zeros((args.batch_size, 1, 224, 224)).cuda()
            predictions_var = torch.zeros((args.batch_size, 1, 224, 224)).cuda()

            for i in range(args.drop_times):
                prediction = model(img_u)[0]
                delta = prediction - predictions_mean
                predictions_mean += delta / (i + 1)
                predictions_var += delta * (prediction - predictions_mean)

            predictions_var /= args.drop_times
            # predictions = [model(img_u)[0] for _ in range(args.drop_times)]
            
            # predictions = torch.stack(predictions)
            # predictions_mean = torch.mean(predictions,dim=0)
            # predictions_var = torch.var(predictions,dim=0)
            possibility = torch.exp(- torch.pow(predictions_var,args.pow))
            # possibility = (predictions_var - predictions_var.min()) / (predictions_var.max() - predictions_var.min() + 1e-8)# (predictions_mean * torch.log(predictions_mean + 1e-8))
            predictions_mean = predictions_mean * possibility
            # possibility = 1 - (predictions_mean * torch.log(predictions_mean + 1e-8))
            # tmp_S1 = -(predictions_mean - 0.5) * (predictions_mean - 0.5)
            # confidence_map = 1 - torch.exp(tmp_S1/(2 * torch.std(predictions_mean) * torch.std(predictions_mean)))
            # threshold = 0.4
            # import pdb;pdb.set_trace()
            # mask = (possibility < threshold).int()

            pred_u = model(img_u)[1] # * possibility
            # predictions_mean = predictions_mean * possibility
            # predictions_mean[1-mask] = 0
            pseudo = (predictions_mean > 0.5)
            pseudo_binary = pseudo.float()
            # pseudo_binary[1-mask] = 0

            loss_u = DeepSupSeg(pred_u,pseudo_binary)
            print("unsup loss:",loss_u)
            loss = 1 * loss_l + 6 * loss_u
            # loss = loss_l
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)
        model.eval()
        if valid_sign == True:
            save_path = os.path.join("./image_output",args.ckpt_name,'expID'+str(args.expID))
            os.makedirs(save_path,exist_ok=True)
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice,name = evaluate(model, valid_dataloader, val_total_batch,save_path)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice))
            
            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if (F1 > F1_best):
                F1_best = F1
                save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_metric': F1,
                    'optimizer': optim.state_dict(),
                }, True,
                    str(epoch),
                    args)
                # torch.save(model.state_dict(), args.root + "/semi/checkpoint/" +'expID'+str(args.expID) +'/'+ args.ckpt_name + "/best.pth")
            
        # else:
        #     recall_test, specificity_test, precision_test, F1_test, F2_test, \
        #     ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test = evaluate(model, test_dataloader, test_total_batch, args)
        #     print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
        #         % (recall_test, specificity_test, precision_test, F1_test, F2_test, ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test))

        #     if (F1_test > F1_test_best):
        #         F1_test_best = F1_test
        #         torch.save(model.state_dict(), args.root + "checkpoint/exp" + str(args.expID) + "/ck_%.4f.pth" % F1_test)




def train_self():
    """load data"""
    train_l_data, train_u_data, valid_data = build_dataset(args)
    train_l_data, train_u_data, valid_data, test_data = build_dataset(args)



    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # test_total_batch = int(len(test_data) / 1)
    """load model"""
    model = build_model(args)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0

    for epoch in range(args.nEpoch):
        model.train()
      
        print("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_u_dataloader), total=total_batch)
        for batch_id, data_u in bar:
            itr = total_batch * epoch + batch_id
            img_u = data_u['image']
            if args.GPUs:
                img_u = img_u.cuda()
            optim.zero_grad()
            gt = model(img_u)
            mask = model(img_u)
            # mask_l = pred_l[:5]
            # inp_l = pred_l[5:]
            loss = DeepSupSeg(mask, gt)
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, valid_dataloader, val_total_batch)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif(F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif(F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")

        # else:
        #     recall_test, specificity_test, precision_test, F1_test, F2_test, \
        #     ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test = evaluate(model, test_dataloader, test_total_batch, args)
        #     print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
        #         % (recall_test, specificity_test, precision_test, F1_test, F2_test, ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test))

        #     if (F1_test > F1_test_best):
        #         F1_test_best = F1_test
        #         torch.save(model.state_dict(), args.root + "checkpoint/exp" + str(args.expID) + "/ck_%.4f.pth" % F1_test)

def test():
  
    print('loading data......')
    test_data = build_dataset(args)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    total_batch = int(len(test_data) / 1)
    model = build_model(args)

    model.eval()
    
    save_path = '/mnt/ai1015/wqc/semi/comparison_image/ProPL/'
    save_path = os.path.join(save_path,args.mode,"expID"+str(args.expID))
    os.makedirs(save_path,exist_ok=True)
    start_time = time.time()
    recall, specificity, precision, F1, F2, \
    ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice, name = evaluate(model, test_dataloader, total_batch,save_path)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Evaluate function runtime: {elapsed_time:.2f} seconds")

    data = {
        "image_name": name,
        "dice": dice
    }
    df = pd.DataFrame(data)

    file_path = args.dataset + "_best.xlsx"
    df.to_excel(file_path, index=False)
    print(f"Excel文件已保存到 {file_path}")

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
        % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean,dice))

    return recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean


if __name__ == '__main__':

    checkpoint_name = os.path.join(args.root, 'semi/checkpoint/'+'expID'+str(args.expID) +'/' + args.ckpt_name)
    if not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)
    else:
        pass
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
    if args.manner == 'full':
        print('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner =='semi':
        print('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        print('---{}-Seg Test---'.format(args.dataset))
        test()
    print('Done')

