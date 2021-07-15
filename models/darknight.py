import torch
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from .discriminator import Discriminator
from .generator import Generator
from .loss import Loss
from ..common import utils as util
from .blocks import *
import os

class DarkNight():
    def __init__(self, opt):
        super(DarkNight, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        opt.Tensor = self.Tensor
        self.gen = Generator(opt)
        self.dis = Discriminator(opt)
        self.setup_networks(opt)
        print('*** Initialized ***')

    def setup_networks(self,opt):
        self.load_network(self.gen, 'G', opt.which_epoch)
        self.load_network(self.dis, 'D', opt.which_epoch)
        self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
        # define loss functions
        self.L1 = torch.nn.SmoothL1Loss()
        self.downsample = torch.nn.AvgPool2d(3, stride=2)
        self.criterionCycle = self.L1
        self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t))
        self.criterionLatent = lambda y,t : self.L1(y, t.detach())
        self.criterionGAN = lambda r,f,v : (Loss(r[0],f[0],v) + \
                                            Loss(r[1],f[1],v) + \
                                            Loss(r[2],f[2],v)) / 3
        # initialize optimizers
        self.gen.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
        self.dis.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
        # initialize loss storage
        self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
        self.loss_cycle = [0]*self.n_domains
        # initialize loss multipliers
        self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
        self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses)])

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        out = self.fc(lstm_out)
        out = self.softmax(out)
        return out
    
    def generator(opt):
        return DarkNight(opt.input_size, opt.hidden_size, opt.output_size)
    
    def discriminator(opt):
        return DarkNight(opt.input_size, opt.hidden_size, 1)
    
    def save_network(self, network, network_label, epoch, gpu_ids):
        save_filename = '%d_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.save(save_path)
        if gpu_ids and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def save(self, label):
        self.save_network(self.gen, 'G', label, self.gpu_ids)
        self.save_network(self.dis, 'D', label, self.gpu_ids)

    def load_network(self, network, network_label, epoch):
        save_filename = '%d_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load(save_path)

    def inference(self, image, domain, dim):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        h, w = image.shape[0], image.shape[1]
        p = transforms.Compose([transforms.Resize((dim, dim))])
        image = p(image)
        p = transforms.ToTensor()
        image = p(image)
        image = image.to("cuda:0")
        self.gen.cuda(0)
        image = image.reshape([1, 3, dim, dim])
        encoded = self.gen.encode(image, domain)
        fake = self.gen.decode(encoded, 1-domain)
        image = fake[0]
        p = transforms.Compose([transforms.Resize((h ,w))])
        image = p(image)
        image_numpy = image.cpu().detach().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        if image_numpy.shape[2] < 3:
            image_numpy = np.dstack([image_numpy]*3)
        imtype=np.uint8
        image_numpy = image_numpy.astype(imtype)
        return image_numpy