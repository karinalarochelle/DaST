# ============================================================================
# DaST: Data Augmentation for Adversarial Training Using Style Transfer
# ============================================================================
# This script trains neural networks to defend against adversarial attacks
# by generating defensive adversarial examples through style transfer.
# 
# Main idea: Train a generator to create diverse adversarial examples that
# help train a more robust discriminator/classifier.
# ============================================================================

from __future__ import print_function
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack  # Fast gradient-based attack method
from sklearn.externals import joblib
# from utils import load_data
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
from net import Net_s, Net_m, Net_l  # Small, Medium, Large networks
from vgg import VGG
from resnet import ResNet50, ResNet18, ResNet34

# Optimize GPU computation
cudnn.benchmark = True

# Create Excel workbook for logging results
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('imitation_network_sig')

# Size of random noise vector fed into the generator
nz = 128

# Check if GPU is available and print status
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# Logger class: writes output to both console and a log file
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream  # Send output to console
        self.log = open(filename, 'a')  # Also write to file

    def write(self, message):
        # Write to both console and log file
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Redirect all print statements to write to log file AND console
sys.stdout = Logger('imitation_network_model.log', sys.stdout)

# Command Line Arguments - Control training with flags
parser = argparse.ArgumentParser()
# Number of parallel workers for loading data
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# How many images per batch during training
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
# Which dataset to use (MNIST, CIFAR, etc.)
parser.add_argument('--dataset', type=str, default='azure')
# How many times to go through training data
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
# Learning rate: how big each gradient update is
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
# Beta1 for Adam optimizer: momentum term
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# Use GPU if available
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
# Optional random seed for reproducibility
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Alpha: weight for diversity loss
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
# Beta: weight for probability matching loss
parser.add_argument('--beta', type=float, default=0.1, help='alpha')
# Generator type (architecture version)
parser.add_argument('--G_type', type=int, default=1, help='iteration limitation')
# Where to save trained model weights
parser.add_argument('--save_folder', type=str, default='saved_model', help='alpha')

# Parse the arguments
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'azure':
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                # normalize,
                                        ]))
    netD = Net_l().cuda()
    netD = nn.DataParallel(netD)

    clf = joblib.load('pretrained/sklearn_mnist_model.pkl')

    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=100, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
        targeted=False)
    nc=1

elif opt.dataset == 'mnist':
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                # normalize,
                                        ]))
    netD = Net_l().cuda()
    netD = nn.DataParallel(netD)

    original_net = Net_m().cuda()
    state_dict = torch.load(
        'pretrained/net_m.pth')
    original_net.load_state_dict(state_dict)
    original_net = nn.DataParallel(original_net)
    original_net.eval()

    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0,
        targeted=False)
    nc=1

data_list = [i for i in range(6000, 8000)] # fast validation
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=opt.workers)
# nc=1

# Set GPU device
device = torch.device("cuda:0" if opt.cuda else "cpu")

# Initialize neural network weights with small random values (Xavier initialization)
# This helps training start better
def weights_init(m):
    classname = m.__class__.__name__
    # For convolutional layers: initialize weights from normal distribution
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # Mean 0, std 0.02
    # For batch norm layers: weights near 1, bias at 0
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Helper function to make predictions using sklearn model on PyTorch tensors
def cal_azure(model, data):
    # Flatten image to 1D vector and move to CPU for sklearn
    data = data.view(data.size(0), 784).cpu().numpy()
    # Get predictions from the model
    output = model.predict(data)
    # Convert back to PyTorch tensor and move to GPU
    output = torch.from_numpy(output).cuda().long()
    return output

# Helper function to get prediction probabilities from sklearn model
def cal_azure_proba(model, data):
    # Flatten image to 1D vector and move to CPU
    data = data.view(data.size(0), 784).cpu().numpy()
    # Get probability predictions (not just class labels)
    output = model.predict_proba(data)
    # Convert back to PyTorch tensor
    output = torch.from_numpy(output).cuda().float()
    return output


# LOSS FUNCTIONS
# Custom loss for training the generator to be more robust
class Loss_max(nn.Module):
    def __init__(self):
        super(Loss_max, self).__init__()

    def forward(self, pred, truth, proba):
        # Loss that combines classification error with probability matching
        criterion_1 = nn.MSELoss()  # Measure if predictions match original model's confidence
        criterion = nn.CrossEntropyLoss()  # Measure classification accuracy
        
        # Convert predictions to probabilities
        pred_prob = F.softmax(pred, dim=1)
        
        # Total loss: classification + probability matching (weighted by beta)
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta
        
        # Convert loss to a value between 0 and 1 (higher is better for training)
        final_loss = torch.exp(loss * -1)
        return final_loss

class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64
        if opt.G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif opt.G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),  # added

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.ReLU(True),

                nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.shape[0]),
                nn.ReLU(True),

                nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False),
                # if self.shape[0] == 3:
                #     nn.Tanh()
                # else:
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.pre_conv(input)
        return output

pre_conv_block = []
for i in range (10):
    pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))

class Generator(nn.Module):
    def __init__(self, num_class):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = num_class
        if opt.G_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 4),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(nc),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
                nn.Sigmoid()
            )
        elif opt.G_type == 2:
            self.main = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True)
            )
    def forward(self, input):
        output = self.main(input)
        return output


# Helper function to split a batch into m smaller chunks
# Example: if batch has 500 images and m=10, returns 10 groups of ~50 images each
def chunks(arr, m):
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]

netG = Generator(10).cuda()
netG.apply(weights_init)
netG = nn.DataParallel(netG)

criterion = nn.CrossEntropyLoss()
criterion_max = Loss_max()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG =  optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

with torch.no_grad():
    correct_netD = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # outputs = netD(inputs)
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, inputs)
        else:
            outputs = original_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_netD += (predicted == labels).sum()
    print('Accuracy of the network on netD: %.2f %%' %
            (100. * correct_netD.float() / total))

################################################
# estimate the attack success rate of initial D:
################################################
correct_ghost = 0.0
total = 0.0
netD.eval()
for data in testloader:
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()

    adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
    with torch.no_grad():
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, adv_inputs_ghost)
        else:
            outputs = original_net(adv_inputs_ghost)
            _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct_ghost += (predicted == labels).sum()
print('Attack success rate: %.2f %%' %
        (100 - 100. * correct_ghost.float() / total))
del inputs, labels, adv_inputs_ghost
torch.cuda.empty_cache()
gc.collect()

batch_num = 100  # Number of iterations to train per epoch (CHANGE THIS to adjust training length)
best_accuracy = 0.0  # Track highest accuracy achieved
best_att = 0.0  # Track best attack success rate

# MAIN TRAINING LOOP - Go through each epoch
for epoch in range(opt.niter):
    # Switch to training mode (enables dropout, batch norm learning)
    netD.train()

    # Inner loop: train for batch_num iterations
    for ii in range(batch_num):
        # Reset gradients to zero
        netD.zero_grad()

        # STEP 1: Update Discriminator (D) - the classifier network
        # Generate random noise to seed the generator
        noise = torch.randn(opt.batchSize, nz, 1, 1, device=device).cuda()
        # Split noise into 10 chunks (one per digit class 0-9)
        noise_chunk = chunks(noise, 10)
        
        # Generate diverse fake images using pre-processor and generator
        for i in range(len(noise_chunk)):
            # Pre-process the noise for class i
            tmp_data = pre_conv_block[i](noise_chunk[i])
            # Generate fake image
            gene_data = netG(tmp_data)
            # Create label i (for diversity loss)
            label = torch.full((noise_chunk[i].size(0),), i).cuda()
            
            # Combine all generated images into one batch
            if i == 0:
                data = gene_data
                set_label = label
            else:
                data = torch.cat((data, gene_data), 0)
                set_label = torch.cat((set_label, label), 0)
        
        # Shuffle the combined data (mix all classes together)
        index = torch.randperm(set_label.size()[0])
        data = data[index]
        set_label = set_label[index]

        # Get predictions from the teacher model (original classifier)
        # These show what the original model thinks about our fake images
        with torch.no_grad():
            if opt.dataset == 'azure':
                # sklearn model - get probabilities
                outputs = cal_azure_proba(clf, data)
                label = cal_azure(clf, data)
            else:
                # PyTorch model
                outputs = original_net(data)
                _, label = torch.max(outputs.data, 1)
                outputs = F.softmax(outputs, dim=1)
            # _, label = torch.max(outputs.data, 1)
        # print(label)

        output = netD(data.detach())
        prob = F.softmax(output, dim=1)
        
        # Calculate two types of loss:
        # 1. Probability loss: discriminator should match original model's confidence
        errD_prob = mse_loss(prob, outputs, reduction='mean')
        # 2. Classification loss: discriminator should classify correctly
        errD_fake = criterion(output, label) + errD_prob * opt.beta
        D_G_z1 = errD_fake.mean().item()  # Save for logging
        
        # Calculate gradients via backpropagation
        errD_fake.backward()

        # Update discriminator weights using optimizer
        errD = errD_fake
        optimizerD.step()

        del output, errD_fake

        # STEP 2: Update Generator (G) - create better fake images
        # Reset gradients for all generator networks
        netG.zero_grad()
        for i in range(10):
            pre_conv_block[i].zero_grad()
        
        # Pass the fake images through discriminator again
        output = netD(data)
        
        # Calculate generator losses:
        # 1. Make images that fool the discriminator (imitation)
        loss_imitate = criterion_max(pred=output, truth=label, proba=outputs)
        # 2. Make images diverse (not all the same within a class)
        loss_diversity = criterion(output, set_label.squeeze().long())
        
        # Combine both losses (alpha controls the balance)
        errG = opt.alpha * loss_diversity + loss_imitate
        
        # If diversity is already good, reduce its weight
        if loss_diversity.item() <= 0.1:
            opt.alpha = loss_diversity.item()
        
        # Calculate gradients for generator
        errG.backward()
        D_G_z2 = errG.mean().item()  # Save for logging
        
        # Update generator weights
        optimizerG.step()
        # Also update the pre-processor networks
        for i in range(10):
            optimizer_block[i].step()

        # Print training progress every 40 iterations
        if (ii % 40) == 0:
            print('[%d/%d][%d/%d] D: %.4f D_prob: %.4f G: %.4f D(G(z)): %.4f / %.4f loss_imitate: %.4f loss_diversity: %.4f'
                % (epoch, opt.niter, ii, batch_num,
                    errD.item(), errD_prob.item(), errG.item(), D_G_z1, D_G_z2, loss_imitate.item(), loss_diversity.item()))


    # End of Epoch: Test the trained discriminator
    # ================================================================
    
    # Test 1: How well does the trained D resist adversarial attacks?
    # (Estimate attack success rate)
    correct_ghost = 0.0
    total = 0.0
    netD.eval()  # Switch to evaluation mode
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Generate adversarial examples (attacks that fool models)
        adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
        with torch.no_grad():
            # See if discriminator can still correctly classify despite attack
            if opt.dataset == 'azure':
                predicted = cal_azure(clf, adv_inputs_ghost)
            else:
                outputs = original_net(adv_inputs_ghost)
                _, predicted = torch.max(outputs.data, 1)
            # _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct_ghost += (predicted == labels).sum()
    print('Attack success rate: %.2f %%' %
            (100 - 100. * correct_ghost.float() / total))
    
    # Save best model based on attack robustness
    if best_att < (total - correct_ghost):
        torch.save(netD.state_dict(),
                    opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
        torch.save(netG.state_dict(),
                    opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
        best_att = (total - correct_ghost)
        print('This is the best model')
    worksheet.write(epoch, 0, (correct_ghost.float() / total).item())
    
    # Clean up memory
    del inputs, labels, adv_inputs_ghost
    torch.cuda.empty_cache()
    gc.collect()

    # Test 2: How accurate is the trained D on regular (non-adversarial) images?
    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        netD.eval()  # Evaluation mode
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # Get predictions from trained discriminator
            outputs = netD(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            
            total += labels.size(0)
            # Count correct predictions
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))
        
        # Save best model based on accuracy
        if best_accuracy < correct_netD:
            torch.save(netD.state_dict(),
                       opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            best_accuracy = correct_netD
            print('This is the best model')
    # Log accuracy to Excel file
    worksheet.write(epoch, 1, (correct_netD.float() / total).item())
workbook.save('imitation_network_saved_azure.xls')

