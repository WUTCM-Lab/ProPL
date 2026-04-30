from .polyp import PolypDataSet
from .skin import SkinDataSet
from .optic import OpticDataSet
from .tn3k import tn3kDataSet
from .BUSI import BUSIDataSet
from .dataset import EgTa,EgTa_one

def build_dataset(args):
    if args.manner == 'test':
        if args.dataset == 'polyp':
            test_data = PolypDataSet(args.root, args.polyp, mode='test')
        elif args.dataset == 'skin':
            test_data = SkinDataSet(args.root, args.skin, mode='test')
        elif args.dataset == 'optic':
            test_data = OpticDataSet(args.root, args.optic, mode='test')
        elif args.dataset == 'tn3k':
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
        elif args.dataset == 'BUSI':
            test_data = BUSIDataSet(args.root,args.expID, mode='test')
        elif args.dataset == 'bfllmott':
            test_data = EgTa_one(args.root,args.bert_type,mode=args.mode)
        return test_data
    else:
        if args.dataset == 'polyp':
            train_data = PolypDataSet(args.root, args.polyp, mode='train', ratio=args.ratio, sign='label')
            valid_data = PolypDataSet(args.root, args.polyp, mode='valid')
            test_data = PolypDataSet(args.root, args.polyp, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = PolypDataSet(args.root, args.polyp, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'skin':
            train_data = SkinDataSet(args.root, args.skin, mode='train', ratio=args.ratio, sign='label')
            valid_data = None
            test_data = SkinDataSet(args.root, args.skin, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = SkinDataSet(args.root, args.skin, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'optic':
            train_data = OpticDataSet(args.root, args.optic, mode='train', ratio=args.ratio, sign='label')
            valid_data = None
            test_data = OpticDataSet(args.root, args.optic, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = OpticDataSet(args.root, args.optic, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'tn3k':
            train_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = tn3kDataSet(args.root, args.expID, mode='valid')
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
            train_u_data = None
            if args.manner == 'semi' or args.manner == 'self':
                train_u_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'BUSI':
            train_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = BUSIDataSet(args.root, args.expID, mode='test')
            # test_data = BUSIDataSet(args.root, args.tn3k, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'bfllmott':
            train_data = EgTa(args.expID,True,args.root,args.bert_type,mode='train')
            train_u_data = EgTa(args.expID,False,args.root,args.bert_type,mode='train')
            valid_data = EgTa(args.expID,True,args.root,args.bert_type,mode='test')
        # return train_data, train_u_data, valid_data, test_data
        return train_data, train_u_data, valid_data


