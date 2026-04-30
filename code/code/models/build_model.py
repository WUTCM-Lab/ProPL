import torch
import models
import os


def build_model(args):
    model = getattr(models, args.model)(args.bert_type, args.vision_type,args.project_dim)
    # model = nn.DataParallel(model)
    if args.GPUs:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    
    # model_dict = model.state_dict()
    # load_ckpt_path = os.path.join('/home/ai1007/code/SemiMedSeg/log/checkpoint/exp2/ck_0.7943.pth')
    # print(load_ckpt_path)
    # print('Loading checkpoint......')
    # checkpoint = torch.load(load_ckpt_path)
    # new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    # print('Done')
    # ckpt_path = os.path.join(args.root, "semi/checkpoint/"+"expID"+str(args.expID)+'/' + str(args.ckpt_name), "best" + '.pth')
    # if os.path.isfile(ckpt_path) and args.load_ckpt is None:
    #     model_dict = model.state_dict()
    #     print("continue learning from ckpt path:",ckpt_path)
    #     checkpoint = torch.load(ckpt_path)
    #     new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    #     model_dict.update(new_dict)
    #     model.load_state_dict(model_dict)
    #     print('Read ckpt Done')



    if args.load_ckpt is not None:

        model_dict = model.state_dict()
        load_ckpt_path = os.path.join(args.root, "semi/checkpoint/"+"expID"+str(args.expID)+'/' + str(args.ckpt_name), args.load_ckpt + '.pth')
        print(load_ckpt_path)
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        ch = torch.load(load_ckpt_path)
        if ch.get('state_dict') is not None:
            ch = torch.load(load_ckpt_path)['state_dict']
        checkpoint = ch
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print('Done')

    return model
