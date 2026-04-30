import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class EgTa(Dataset):

    def __init__(self, exp_ID,is_label, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(EgTa, self).__init__()

        self.mode = mode
        self.is_label = is_label
        classes_num = os.listdir(root_path)
        image_path_lists = []
        # import pdb;pdb.set_trace()
        if mode == 'train':
            if exp_ID == 1:
                label_ratio = 0.125
            elif exp_ID == 2:
                label_ratio = 0.0625
            elif exp_ID == 3:
                label_ratio = 0.25

            if is_label:
                for cls in classes_num:
                    if cls == "UniSeg" or cls == "semi":
                        continue
                    each_cls_path = os.path.join(root_path,cls)
                    image_path = os.path.join(each_cls_path,mode,'img')
                    imgs = [os.path.join(image_path,img) for img in os.listdir(image_path)]
                    image_path_lists += imgs[0:int(len(imgs)*label_ratio)]
            else:
                for cls in classes_num:
                    if cls == "UniSeg" or cls == "semi":
                        continue
                    each_cls_path = os.path.join(root_path,cls)
                    image_path = os.path.join(each_cls_path,mode,'img')
                    imgs = [os.path.join(image_path,img) for img in os.listdir(image_path)]
                    image_path_lists += imgs[int(len(imgs)*label_ratio):]
        else:
            for cls in classes_num:
                if cls == "UniSeg" or cls == "semi":
                    continue
                each_cls_path = os.path.join(root_path,cls)
                image_path = os.path.join(each_cls_path,mode,'img')
                imgs = [os.path.join(image_path,img) for img in os.listdir(image_path)]
                image_path_lists += imgs
        self.image_list = image_path_lists
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        print("length of dataset",len(self.image_list))
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = self.image_list[idx]
        gt = self.image_list[idx].replace('jpg', 'png').replace('img','mask')
        task = image.split('/')[6].replace('_',' ').lower()
        caption = "Segment the {} in the ultrasound image.".format(task)
        # caption = ""

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']
        # caption = self.caption_list[idx]

        # token_output = self.tokenizer.encode_plus(caption, padding='max_length',
        #                                                 max_length=24, 
        #                                                 truncation=True,
        #                                                 return_attention_mask=True,
        #                                                 return_tensors='pt')
        # token,mask = token_output['input_ids'],token_output['attention_mask']
 
        # data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = {'image':image, 'gt':gt, 'token':token}
        data = trans(data)

        image,gt,token = data['image'],data['gt'],data['token']
        gt = torch.where(gt>0,1,0)
        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        # print("task:",task)
        # print("token",token)

        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        if gt.shape[0] == 1:
            gt = gt.repeat(3,1,1)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}
        # text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        # text = {'input_ids':token, 'attention_mask':mask} 

        if not self.is_label and self.mode == 'train':
            return ([image,text])
            
        
        return ([image, text], gt)
        

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token"]),

            ])

        return trans
    

class EgTa_one(Dataset):

    def __init__(self, root_path=None, tokenizer=None, mode='brec',image_size=[224,224]):

        super(EgTa_one, self).__init__()


        self.mode = mode
        
        image_path_lists = []
        
        each_cls_path = os.path.join(root_path,mode)
        image_path = os.path.join(each_cls_path,'test','img')
        imgs = [os.path.join(image_path,img) for img in os.listdir(image_path)]
        image_path_lists = imgs
        self.image_list = image_path_lists
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = self.image_list[idx]
        img_path = self.image_list[idx]
        gt = self.image_list[idx].replace('jpg', 'png').replace('img','mask')
        task = image.split('/')[6].replace('_',' ').lower()
        #  print(task)
        caption = "Segment the {} in the ultrasound image.".format(task)
        # caption = ""

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']
        # caption = self.caption_list[idx]

        # token_output = self.tokenizer.encode_plus(caption, padding='max_length',
        #                                                 max_length=24, 
        #                                                 truncation=True,
        #                                                 return_attention_mask=True,
        #                                                 return_tensors='pt')
        # token,mask = token_output['input_ids'],token_output['attention_mask']
 
        # data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = {'image':image, 'gt':gt, 'token':token}
        data = trans(data)

        image,gt,token = data['image'],data['gt'],data['token']
        gt = torch.where(gt>0,1,0)
        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        # print("task:",task)
        # print("token",token)

        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        if gt.shape[0] == 1:
            gt = gt.repeat(3,1,1)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}
        # text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        # text = {'input_ids':token, 'attention_mask':mask} 
        case_name = task + img_path.split('/')[-1].split('.')[0]

        return ([image, text], gt,case_name)


    def transform(self,image_size=[224,224]):

        trans = Compose([
            LoadImaged(["image","gt"], reader='PILReader'),
            EnsureChannelFirstd(["image","gt"]),
            Resized(["image"],spatial_size=image_size,mode='bicubic'),
            Resized(["gt"],spatial_size=image_size,mode='nearest'),
            NormalizeIntensityd(['image'], channel_wise=True),
            ToTensord(["image","gt","token"]),

        ])

        return trans