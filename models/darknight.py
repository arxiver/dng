import torch
from .discriminator import Discriminator
from .generator import Generator
from .loss import Loss

class DarkNight(torch.nn.Module):
    def __init__(self, opt):
        super(DarkNight, self).__init__()
    
    
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
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

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
        self.netG.cuda(0)
        image = image.reshape([1, 3, dim, dim])
        encoded = self.netG.encode(image, domain)
        fake = self.netG.decode(encoded, 1-domain)
        image = fake[0]
        p = transforms.Compose([transforms.Resize((h ,w))])
        image = p(image)
        image_numpy = img.cpu().detach().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        plt.imshow(image_numpy)
        if image_numpy.shape[2] < 3:
            image_numpy = np.dstack([image_numpy]*3)
        imtype=np.uint8
        image_numpy = image_numpy.astype(imtype)
        return image_numpy