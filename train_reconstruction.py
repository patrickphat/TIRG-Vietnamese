from img_text_composition_models import TIRGReconstructionModel
import img_text_composition_models
from enum import Enum
import datasets
import torchvision
import torch
from main import load_dataset, create_model_and_optimizer
import test_retrieval
from tqdm import tqdm
import cv2
from utils.FilePickling import pkl_save, pkl_load
import numpy as np
from torch.nn import MSELoss
from collections import OrderedDict
from torch.optim import SGD


class opt_config:
    def __init__(self):
        self.dataset = "css3d"
        self.dataset_path = "../data/CSSDataset/CSS-vn-vanilla-v2.json" 
        self.model = "tirg" 
        self.loss = "soft_triplet" 
        self.comment = "css3d_tirg"
        self.embed_dim = 512
        self.learning_rate = 1e-2
        self.weight_decay = 1e-6 
        self.f = ""
        self.learning_rate_decay_frequency = 99999999
        self.batch_size = 32
        self.num_epochs = 100
        self.n_epochs_valudations = 5
        self.loader_num_workers = 4
        self.pretrained_weights = "runs/Sep23_05-38-54_ai-servers-3css_vn_vanilla_v2/latest_checkpoint.pth"

opt = opt_config()

if __name__ == "__main__":
        
    opt = opt_config()

    trainset, testset = load_dataset(opt)
    texts = [t for t in trainset.get_all_texts()]
    model, optimizer = create_model_and_optimizer(opt, [t for t in trainset.get_all_texts()])    
    rec_model = TIRGReconstructionModel(model, embed_dim=512, img_shape = [3,120,180]).cuda()
    
    pretrained_weights = torch.load("best_rec_model.pth")
    rec_model.load_state_dict(pretrained_weights)

    trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)

    testloader = testset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)

    criteria =  MSELoss(reduction = 'mean')
    n_batches = len(trainloader)
    n_epochs = 100
    n_epochs_valid = 3

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, rec_model.parameters()), lr=0.1, momentum=0.9)

    progressbar = tqdm(
            range(n_batches),
            total=n_batches,
            desc=f"[Training]::",
        )

    best_loss = float("inf")

    for i in range(n_epochs):
        progressbar = tqdm(
                range(n_batches),
                total=n_batches,
                desc=f"[Training]::",
            )

        train_data = iter(trainloader)
        losses = []
        for t in progressbar:
            data = next(train_data)
            img1 = np.stack([d['source_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float()
            img1 = torch.autograd.Variable(img1).cuda()
            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float()
            img2 = torch.autograd.Variable(img2).cuda()
        #     import ipdb
        #     ipdb.set_trace()
            mods = [str(d['mod']['str']) for d in data]
            mods = [t for t in mods]

            rec_img = rec_model.forward(img1,mods)
            loss = criteria(rec_img, img2)
            loss_detach = loss.data.cpu().numpy().item()
            losses.append(loss_detach)

            postfix_progress = OrderedDict()
            postfix_progress["L2_loss"] = loss_detach
            
            loss.backward()
        #         import ipdb
        #         ipdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()
            
            progressbar.set_postfix(ordered_dict=postfix_progress)

        m_loss = np.mean(losses)
        if m_loss < best_loss:
            best_loss = m_loss
            torch.save(rec_model.state_dict(), "best_rec_model.pth")
            print(f"Saved model with mean accu loss = {m_loss}")
