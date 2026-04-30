import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
from utils.aug_function import DropOutDecoder
import torchvision.models as models

class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1 + 2 + last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load("/home/ai1015/wqc/semi/MyModel/code/models/pretrain/backbone/resnet34.pth"))

        if in_channels == 3:
            self.encoder1_conv = resnet.conv1
        else:
            self.encoder1_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_maxpool = self.maxpool(e1)

        e2 = self.encoder2(e1_maxpool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5




class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        print("project_dim:",project_dim)
        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]
        # feature_dim = [512,256,128,64]
        #  import pdb;pdb.set_trace()
        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.uns_decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.uns_decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.uns_decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.uns_decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.uns_out = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.dropout = DropOutDecoder()

    def forward(self, data,is_sup=True):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 
        
        
        #  import pdb; pdb.set_trace()
        os32 = self.dropout(image_features[3])

        os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        # import pdb; pdb.set_trace()
        os1 = self.decoder1(os4)

        mask = self.out(os1).sigmoid()
        
    

        uns32 = image_features[3]
        uns16 = self.uns_decoder16(uns32,image_features[2], text_embeds[-1])
        uns8 = self.uns_decoder8(uns16,image_features[1], text_embeds[-1])
        uns4 = self.uns_decoder4(uns8,image_features[0], text_embeds[-1])
        uns4 = rearrange(uns4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        uns1 = self.uns_decoder1(uns4)

        pred_bond = self.uns_out(uns1).sigmoid()
        return mask,pred_bond
# class LanGuideMedSeg(nn.Module):

#     def __init__(self, bert_type, vision_type, project_dim=512):

#         super(LanGuideMedSeg, self).__init__()

#         # self.encoder = VisionModel(vision_type, project_dim)
#         self.encoder = Encoder(3)
#         self.text_encoder = BERTModel(bert_type, project_dim)
#         print("project_dim:",project_dim)
#         self.spatial_dim = [7,14,28,56,56]    # 224*224
#         feature_dim = [512,256,128,64,64]
#         # feature_dim = [512,256,128,64]
#         #  import pdb;pdb.set_trace()
#         self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
#         self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
#         self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
#         self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
#         self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

#         self.uns_decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
#         self.uns_decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
#         self.uns_decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
#         self.uns_decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
#         self.uns_out = UnetOutBlock(2, in_channels=24, out_channels=1)

#         self.dropout = DropOutDecoder()

#     def forward(self, data,is_sup=True):

#         image, text = data
#         if image.shape[1] == 1:   
#             image = repeat(image,'b 1 h w -> b c h w',c=3)

#         e1,e2,e3,e4,e5 = self.encoder(image)
        
#         text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
#         text_embeds, text_project = text_output['feature'],text_output['project']

#         if len(image_features[0].shape) == 4: 
#             image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
#             image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 
        
        
#         #  import pdb; pdb.set_trace()
#         os32 = image_features[3]

#         os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
#         os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
#         os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
#         os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
#         os1 = self.decoder1(os4)

#         mask = self.out(os1).sigmoid()
        
#         pseudo = (mask > 0.5)
#         pseudo_binary = pseudo.float()

#         uns32 = self.dropout(image_features[3])
#         uns16 = self.uns_decoder16(uns32,image_features[2], text_embeds[-1])
#         uns8 = self.uns_decoder8(uns16,image_features[1], text_embeds[-1])
#         uns4 = self.uns_decoder4(uns8,image_features[0], text_embeds[-1])
#         uns4 = rearrange(uns4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
#         uns1 = self.uns_decoder1(os4)

#         pred_bond =  self.uns_out(uns1).sigmoid()
#         return mask,pred_bond,pseudo_binary
