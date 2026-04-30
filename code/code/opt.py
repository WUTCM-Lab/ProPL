import argparse
import os


parse = argparse.ArgumentParser(description='PyTorch Semi-Medical-Seg Implement')

"-------------------GPU option----------------------------"
parse.add_argument('--GPUs', type=str, default='0')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/home/ai1015/wqc/semi/wyjsemi')
parse.add_argument('--dataset', type=str, default='polyp', choices=['polyp', 'skin', 'optic','tn3k','BUSI','bfllmott'])
parse.add_argument('--ratio', type=int, default=10)
parse.add_argument('--polyp', type=str, default='data_polyp')
parse.add_argument('--skin', type=str, default='data_skin')
parse.add_argument('--optic', type=str, default='data_optic')
parse.add_argument('--tn3k', type=str, default='tn3k')
parse.add_argument('--BUSI', type=str, default='BUSI')
# bert_type: ./lib/BiomedVLP-CXR-BERT-specialized
#   vision_type: ./lib/convnext-tiny-224
parse.add_argument('--bert_type',type=str,default='/home/ai1015/wqc/semi/MyModel/lib/BiomedVLP-CXR-BERT-specialized')
parse.add_argument('--vision_type',type=str,default='/home/ai1015/wqc/semi/MyModel/lib/convnext-tiny-224')
# project_dim: 768
parse.add_argument('--project_dim',type=int,default=768)
"-------------------training option-----------------------"
parse.add_argument('--manner', type=str, default='full', choices=['full', 'semi', 'test', 'self'])
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=200)
parse.add_argument('--batch_size', type=int, default=24)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--model', type=str, default='LanGuideMedSeg')
parse.add_argument('--expID', type=int, default=0) 
parse.add_argument('--ckpt_name', type=str, default=None)
parse.add_argument('--drop_times',type=int,default=5)
parse.add_argument('--resume',type=str)

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--power',type=float, default=0.9)
parse.add_argument('--betas', default=(0.9, 0.999))
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--eps', type=float, default=1e-8)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--pow', type=int, default=2)

parse.add_argument('--nclasses', type=int, default=1)
parse.add_argument('--backbone', type=str, default='tiny')
parse.add_argument('--band', type=int, default=3)
parse.add_argument('--save_pred', default=False, action='store_true')

args = parse.parse_args()
