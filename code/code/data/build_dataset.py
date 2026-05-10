from .dataset import EgTa,EgTa_one

def build_dataset(args):
    if args.manner == 'test':
        if args.dataset == 'bfllmott':
            test_data = EgTa_one(args.root,args.bert_type,mode=args.mode)
        return test_data
    else:
        if args.dataset == 'bfllmott':
            train_data = EgTa(args.expID,True,args.root,args.bert_type,mode='train')
            train_u_data = EgTa(args.expID,False,args.root,args.bert_type,mode='train')
            valid_data = EgTa(args.expID,True,args.root,args.bert_type,mode='test')
        # return train_data, train_u_data, valid_data, test_data
        return train_data, train_u_data, valid_data


