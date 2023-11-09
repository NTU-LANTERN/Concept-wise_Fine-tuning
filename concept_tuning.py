import torch
import torch.nn as nn
from torch.nn.functional import normalize
from vision.models.classifier import Classifier as ClassifierBase
from vision.models.classifier import DANet
from torchvision.ops import roi_align
import torch.nn.functional as F
import numpy as np
import copy
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
from utils import SinkhornDistance, fix_bn, recover_bn
import time
# from einops import rearrange, repeat, reduce
# features = roi_align(feature_maps, boxes.cuda().float(), output_size=(2, 2), spatial_scale=1/32, aligned=True

class Classifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck=None, ib_dim=512, finetune=True):
        head = nn.Linear(backbone.out_features, num_classes)
        head.weight.data.normal_(0, 0.01)
        head.bias.data.fill_(0.0)
        super(Classifier, self).__init__(backbone, num_classes=num_classes, bottleneck=bottleneck, head=head, finetune=finetune)
        self.backbone_features = backbone.out_features
        self.ib_dim = ib_dim
        self.IB_encoder = nn.Sequential(
            DANet(2048), ###Ateention model
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),
            nn.Linear(2048, 2 * self.ib_dim)) 
        self.IB_decoder = nn.Linear(self.ib_dim, num_classes)
    
    def forward(self, x, info=False):
        f = self.backbone(x)
        h = self.bottleneck(f)
        y = self.head(h)
        logit, mu, std = self.IB(f)
        if not info:
            return y, h, f, logit
        else:
            return y, h, f, logit, mu, std

    def IB(self, x):
        statistics = self.IB_encoder(x)
        dim =  statistics.shape[-1]//2
        mu = statistics[:,:dim]
        std = F.softplus(statistics[:,dim:]-5,beta=1)
        encoding = self.reparametrize_n(mu,std)
        logit = self.IB_decoder(encoding)
        return logit, mu, std

    def reparametrize_n(self, mu, std, n=1):
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())
        if n != 1 :
            mu = expand(mu)
            std = expand(std)
        eps = Variable(std.data.new(std.size()).normal_().cuda())
        return mu + eps * std

    def get_parameters(self, base_lr=1e-2):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.IB_encoder.parameters(),"lr": 1.0 * base_lr},
            {"params": self.IB_decoder.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

class Contuning(nn.Module):
    def __init__(self, encoder_q: Classifier, encoder_k: Classifier, num_classes, K=40, m=0.999, T=0.07, confusing=100):
        super(Contuning, self).__init__()
        self.K = K
        self.K_patch = K
        self.m = m
        self.T = T
        self.num_classes = num_classes
        self.local_number = 9
        self.channel_number = 30
        self.patch_size = 64
        self.confusing = confusing
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue for image features
        self.register_buffer("queue_z", torch.randn(num_classes, num_classes, K))
        self.queue_z = normalize(self.queue_z, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        # create the queue for concept features
        self.register_buffer("queue_h", torch.randn(num_classes, self.K_patch, self.local_number, self.num_classes))
        self.queue_h = normalize(self.queue_h, dim=-1)
        self.register_buffer("queue_h_ptr", torch.zeros(num_classes, dtype=torch.long))
        
        ## Loss function
        self.sinkhorn = SinkhornDistance(eps=1e-2, max_iter=200, reduction=None).cuda()
        self.contrastive_criterion = torch.nn.KLDivLoss(reduction='batchmean')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z, h, label):
        batch_size = z.shape[0]
        # assert self.K % batch_size == 0  # for simplicity
        ptr = int(self.queue_ptr[label])
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_z[:, label, ptr: ptr+batch_size] = z.T
        # move pointer
        self.queue_ptr[label] = (ptr + batch_size) % self.K

        ptr = int(self.queue_h_ptr[label])
        self.queue_h[label, ptr: ptr+batch_size, :, :] = h
        self.queue_h_ptr[label] = (ptr + batch_size) % self.K_patch

    def generate_box(self, attentive_maps,index):
        # (x1, y1, x2, y2)
        lt = self.patch_size//2
        rb = self.patch_size//2
        box = []
        for i,attentive_map in enumerate(attentive_maps):
            peak = torch.argmax(attentive_map)
            y = (peak//7)*32
            x = (peak%7)*32
            box.append(torch.tensor([index,max(0,x-lt),max(0,y-lt), min(224,x+rb),min(224,y+rb)]))
        box = torch.stack(box)
        return box.cuda().float()
    
    ### search concept patches
    def _search_patches(self, feature_maps, preds, labels):
        batch_size = feature_maps.size()[0]
        model_output = preds.detach().cpu().numpy()
        label = labels.cpu().numpy()
        cls_weights = copy.deepcopy(self.encoder_k.head).weight.data
        boxes = []
        f = feature_maps.clone().detach()
        confusing_classes_all = []
        for i in range(batch_size):
            atten = f[i]
            confusing_classes = (-model_output[i]).argsort()[:self.confusing+1]
            confusing_classes = np.delete(confusing_classes, np.where(confusing_classes == label[i]))[:self.confusing]
            confusing_classes_all.append(confusing_classes)
            target_score = 3*cls_weights[label[i]]-cls_weights[confusing_classes[0]] \
                -cls_weights[confusing_classes[1]]-cls_weights[confusing_classes[2]]
            _, all_index = torch.topk(target_score, 2048)
            ini = np.arange(0,2048,250)
            attentive_maps = []
            for in_index in ini:
                attentive_maps.append(torch.sum(atten[all_index[in_index:in_index+self.channel_number]],dim=0, keepdim=False))
            attentive_maps.append(torch.sum(atten[all_index[-self.channel_number:]],dim=0, keepdim=False))
            boxes.append(self.generate_box(attentive_maps[:self.local_number], i))
        boxes = torch.cat(boxes)
        return boxes, confusing_classes_all

    ### encode concept patches (Time-consuming, can be replaced with RoIcrop on the encoded image feature maps)
    def patch_features(self, images, boxes, q=True):
        patch_emb = []
        batch_size = images.size(0)
        patches = roi_align(images, boxes, output_size=(self.patch_size, self.patch_size), spatial_scale=1.0, aligned=True)
        patches = patches.reshape(batch_size, self.local_number, 3, self.patch_size, self.patch_size)
        if q:
            class_cls = copy.deepcopy(self.encoder_q.head)
            for i in range(self.local_number ):
                patch = patches[:,i,:,:,:].contiguous()
                idx_shuffle = torch.randperm(batch_size).cuda()
                _, patch_rep,_,_ = self.encoder_q(patch[idx_shuffle])
                patch_rep = normalize(class_cls(patch_rep),dim=1)
                idx_unshuffle = torch.argsort(idx_shuffle)
                patch_emb.append(patch_rep[idx_unshuffle].unsqueeze(1))
        else:
            class_cls = copy.deepcopy(self.encoder_k.head)
            for i in range(self.local_number):
                patch = patches[:,i,:,:,:].contiguous()
                idx_shuffle = torch.randperm(batch_size).cuda()
                _, patch_rep,_,_ = self.encoder_k(patch[idx_shuffle])
                patch_rep = normalize(class_cls(patch_rep),dim=1)
                idx_unshuffle = torch.argsort(idx_shuffle)
                patch_emb.append(patch_rep[idx_unshuffle].unsqueeze(1))
        patch_emb = torch.cat(patch_emb, axis=1)
        return patch_emb

    def forward(self, imgs, labels):
        batch_size = imgs[0].size(0)
        device = imgs[0].device

        self.encoder_q.apply(recover_bn) #
        y_q, _, _, ib_preds, mu, std = self.encoder_q(imgs[0], True)
        z_q = normalize(y_q, dim=1) #image feature for contrastive learning
        
        self.encoder_q.apply(fix_bn) # If encoding the concept patches, avoid BN overfiting
        y_q_2, _, feature_q,_ = self.encoder_k(imgs[2])
        boxes_q, confusing_classes_all = self._search_patches(feature_q,y_q_2,labels)
        z_q_p = self.patch_features(imgs[2],boxes_q) #concept feature for contrastive learning

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y_k, _, _,_= self.encoder_k(imgs[1])
            z_k = normalize(y_k,dim=1)

            y_k_2, _, feature_k,_ = self.encoder_k(imgs[3])
            boxes_k, _ = self._search_patches(feature_k,y_k_2,labels)
            z_k_p = self.patch_features(imgs[3], boxes_k,False)

        # compute logits for projection z
        # Image-level
        logits_z_cur = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)
        queue_z = self.queue_z.clone().detach().to(device)
        # positive logits: N x K
        logits_z_pos = torch.Tensor([]).to(device)
        # negative logits: N x ((C-1) x K)
        logits_z_neg = torch.Tensor([]).to(device)

        # concept-level
        logits_z_cur_concept = 1.0-self.sinkhorn(z_q_p,z_k_p,True).unsqueeze(-1)
        queue_h = self.queue_h.clone().detach().to(device)
        logits_z_pos_concept  = torch.Tensor([]).to(device)
        logits_z_neg_concept = torch.Tensor([]).to(device)

        for i in range(batch_size):
            c = labels[i]
            pos_samples = queue_z[:, c, :]  
            neg_samples = torch.cat([queue_z[:, 0: c, :], queue_z[:, c+1:, :]], dim=1).flatten(start_dim=1)  
            ith_pos = torch.einsum('nc,ck->nk', [z_q[i: i+1], pos_samples])  
            ith_neg = torch.einsum('nc,ck->nk', [z_q[i: i+1], neg_samples])   
            logits_z_pos = torch.cat((logits_z_pos, ith_pos), dim=0)
            logits_z_neg = torch.cat((logits_z_neg, ith_neg), dim=0)

            confusing_classes = confusing_classes_all[i]
            pos_samples_h = queue_h[c,:, :, :]  
            neg_samples_h = queue_h[confusing_classes].flatten(end_dim=1)   
            ith_pos_concept = 1.-self.sinkhorn(z_q_p[i: i+1],pos_samples_h, True).unsqueeze(0)
            logits_z_pos_concept = torch.cat((logits_z_pos_concept, ith_pos_concept), dim=0)
            ith_neg_concept = 1.-self.sinkhorn(z_q_p[i: i+1],neg_samples_h, False).unsqueeze(0)
            logits_z_neg_concept = torch.cat((logits_z_neg_concept, ith_neg_concept), dim=0)

            self._dequeue_and_enqueue(z_k[i:i+1], z_k_p[i:i+1], labels[i])

        # calculate losses
        logits_z = torch.cat([logits_z_cur, logits_z_pos, logits_z_neg], dim=1) 
        logits_z = nn.LogSoftmax(dim=1)(logits_z/self.T)

        logits_z_concept = torch.cat([logits_z_cur_concept, logits_z_pos_concept, logits_z_neg_concept], dim=1) 
        logits_z_concept = nn.LogSoftmax(dim=1)(logits_z_concept/ self.T)

        # image level contrastive labels
        labels_c = torch.zeros([batch_size, self.K * self.num_classes+1]).to(device)
        labels_c[:, 0:self.K+1].fill_(1.0 / (self.K+1))

        # patch level contrastive labels
        labels_c_concept = torch.zeros([batch_size, (self.confusing*10+1) * self.K_patch + 1]).to(device)
        labels_c_concept[:, 0:self.K_patch+1].fill_(1.0 / (self.K_patch + 1))


        contrastive_loss = self.contrastive_criterion(logits_z, labels_c)
        conncept_loss = self.contrastive_criterion(logits_z_concept, labels_c_concept)

        normal_prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
        latent_prior = Independent(Normal(loc=mu, scale=torch.exp(std)), 1)
        IB_info_loss = torch.mean(kl.kl_divergence(latent_prior, normal_prior))

        return y_q, contrastive_loss, conncept_loss, ib_preds, IB_info_loss
        
