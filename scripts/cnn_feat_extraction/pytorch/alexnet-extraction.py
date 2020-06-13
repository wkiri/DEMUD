import sys, os
import numpy as np
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

def usage():
    print("usage: python ex-model.py dataset_dir out_dir layer batch")
    sys.exit(1)

if __name__ == "__main__":

    if len(sys.argv) != 5:
        usage()
    
    # read dataset directory
    img_dir = sys.argv[1]
    if not os.path.isdir(out_dir):
        print("error: {} is not an existing directory".format(img_dir))
        sys.exit(1)

    # read output directory
    out_dir = sys.argv[2]
    if not os.path.isdir(out_dir):
        print("info: creating directory {}".format(out_dir))
        os.mkdir(out_dir)

    # read layer
    layer = sys.argv[3]
    if layer not in ['fc6', 'fc7', 'fc8']:
        print("error: {} is not a valid layer".format(layer))
        sys.exit(1)

    # read batch size
    try:
        BATCH = int(sys.argv[4])
    except:
        print("error: {} is not an integer".format(sys.argv[4])
        sys.exit(1)
    
    
    # preprocessing definitions
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    
    # define model
    model = models.alexnet(pretrained=True)
    model.cuda()    # delete all lines with .cuda() if you don't have CUDA
    model.eval()

    # register hooks for feature extraction
    fc6 = model._modules.get('classifier')[1]
    fc7 = model._modules.get('classifier')[4]

    fc6_buf = torch.zeros(BATCH,4096).cuda()
    fc7_buf = torch.zeros(BATCH,4096).cuda()

    def fc6_hook(m, i, o):
        if o.data.shape[0] != BATCH:
            temp = torch.zeros(BATCH - o.data.shape[0], 4096).cuda()
            temp = torch.cat((o.data, temp))
            fc6_buf.copy_(temp)
        else:
            fc6_buf.copy_(o.data)
    def fc7_hook(m, i, o):
        if o.data.shape[0] != BATCH:
            temp = torch.zeros(BATCH - o.data.shape[0], 4096).cuda()
            temp = torch.cat((o.data, temp))
            fc7_buf.copy_(temp)
        else:
            fc7_buf.copy_(o.data)

    fc6.register_forward_hook(fc6_hook)
    fc7.register_forward_hook(fc7_hook)

    # set up pytorch dataloader for images
    dataset = datasets.ImageFolder(root=img_dir, transform=img_transform)
    dataset_loader = data.DataLoader(dataset, batch_size=BATCH)

    # extract features
    fc6_out = []
    fc7_out = []
    fc_out = []
    for i, (img_batch, _) in enumerate(dataset_loader,0):
        print("Extracting batch {}".format(i))
        img_batch = img_batch.cuda()
        ptr = i * BATCH

        fc_buf = model(img_batch)

        if layer == 'fc6':
            fc6_list = fc6_buf.tolist()
        elif layer == 'fc7':
            fc7_list = fc7_buf.tolist()
        elif layer == 'fc8':
            fc_list = fc_buf.tolist()

        for j in range(len(fc_buf)):
            img_name = os.path.split(dataset.imgs[ptr+j][0])[-1]
            if layer == 'fc6':
                fc6_out.append([img_name] + fc6_list[j])
            elif layer == 'fc7':
                fc7_out.append([img_name] + fc7_list[j])
            elif layer == 'fc8':
                fc_out.append([img_name] + fc_list[j])
    
    # save features to csv 
    if layer == 'fc6':
        with open(os.path.join(out_dir, "fc6.csv"), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(fc6_out)
    elif layer == 'fc7':
        with open(os.path.join(out_dir, "fc7.csv"), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(fc7_out)
    elif layer == 'fc8':
        with open(os.path.join(out_dir, "fc.csv"), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(fc_out)

