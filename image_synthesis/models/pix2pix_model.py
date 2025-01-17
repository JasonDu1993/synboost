import torch
import models.networks as networks
import util.util as util
import pdb
from time import time
import torch.nn.functional as F


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor
        self.BoolTensor = torch.cuda.BoolTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
        torch.manual_seed(1234)
        self.eps = torch.randn((1, 256))

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        if isinstance(mode, str):
            input_semantics, real_image = self.preprocess_input(data)

            if mode == 'generator':
                g_loss, generated = self.compute_generator_loss(
                    input_semantics, real_image)
                return g_loss, generated
            elif mode == 'discriminator':
                d_loss = self.compute_discriminator_loss(
                    input_semantics, real_image)
                return d_loss
            elif mode == 'encode_only':
                z, mu, logvar = self.encode_z(real_image)
                return mu, logvar
            elif mode == 'inference':
                with torch.no_grad():
                    fake_image, _ = self.generate_fake(input_semantics, real_image)
                return fake_image
            else:
                raise ValueError("|mode| is invalid")
        else:
            # for modularity, I use the same name variables but this is used for onnx converion
            label = data
            image = mode
            label = label.cuda(self.opt.gpu)
            image = image.cuda(self.opt.gpu)

            with torch.no_grad():
                fake_image, _ = self.generate_fake(label, image)
            return fake_image

    # Deprecated! (Corrected by If/Else statement in forward function)
    def _forward(self, label, image):
        # move to GPU and change data types
        label = label.cuda(self.opt.gpu)
        image = image.cuda(self.opt.gpu)

        with torch.no_grad():
            fake_image, _ = self.generate_fake(label, image)
        return fake_image

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        if "mask" in data:
            mask = data["mask"]
        else:
            data["label"] = data["label"].long()
            data['label'] = data['label'].cuda(self.opt.gpu)
            data['instance'] = data['instance'].cuda(self.opt.gpu)
            data['image'] = data['image'].cuda(self.opt.gpu)

            # create one-hot label map
            label_map = data["label"]
            bs, _, h, w = label_map.size()  # shape [bs, 1, h, w]
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = torch.zeros(bs, nc, h, w).to(label_map.device)
            input_semantics = input_label.scatter_(1, label_map, 1.0)
            mask = input_semantics

        # print(torch.sum(mask!=input_semantics1))

        """ 替换方案1：会产生inplace操作
        input_label = torch.zeros(bs * nc * h * w).to(label_map.device)
        label_map = label_map.flatten()

        e = torch.range(0, bs * nc * h * w - 1, nc).to(label_map.device).long()
        label_map = label_map + e
        # input_label[label_map.flatten()] = 1
        # input_label[1] = 1
        torch.cuda.synchronize()
        t0 = time()
        for i in range(bs * h * w):
            input_label[label_map[i]] = 1
        torch.cuda.synchronize()
        print("img syn_net scatter Equivalent {} s".format(time() - t0))
        input_label = input_label.reshape(bs, h, w, nc).permute(0, 3, 1, 2)
        """

        # torch.cuda.synchronize()
        # t0 = time()
        # for i in range(bs):
        #     for j in range(h):
        #         for k in range(w):
        #             ind = label_map[i, 0, j, k]
        #             input_label[i, ind, j, k] = 1
        # torch.cuda.synchronize()
        # print("img syn_net scatter Equivalent {} s".format(time() - t0))

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((mask, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        feat_fake, pred_fake, feat_real, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)
        if not self.opt.no_ganFeat_loss:
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            num_D = len(feat_fake)
            for i in range(num_D):
                GAN_Feat_loss += self.criterionFeat(
                    feat_fake[i], feat_real[i].detach()) * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        _, pred_fake, _, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_and_real_img = torch.cat([fake_image, real_image], dim=0)

        discriminator_out = self.netD(fake_and_real_img,
                                      segmap=torch.cat((input_semantics, input_semantics), dim=0))

        fake_feats, fake_preds, real_feats, real_preds = self.divide_pred(discriminator_out)

        return fake_feats, fake_preds, real_feats, real_preds

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        fake_feats = []
        fake_preds = []
        real_feats = []
        real_preds = []
        for p in pred[0]:
            fake_feats.append(p[:p.size(0) // 2])
            real_feats.append(p[p.size(0) // 2:])
        for p in pred[1]:
            fake_preds.append(p[:p.size(0) // 2])
            real_preds.append(p[p.size(0) // 2:])

        return fake_feats, fake_preds, real_feats, real_preds

    def get_edge1(self, x, t, pad):
        a = (t[:, :, :, 1:] != t[:, :, :, :-1]).float()
        a = F.pad(a, pad, mode="constant", value=0)
        t = (x + a)
        t[t > 1] = 1
        return t

    def get_edge2(self, x, t, pad):
        a = (t[:, :, 1:, :] != t[:, :, :-1, :]).float()
        a = F.pad(a, pad, mode="constant", value=0)
        t = (x + a)
        t[t > 1] = 1
        return t

    def get_edges(self, t):
        # print("t", t.shape)
        # edge = torch.zeros_like(t, dtype=torch.bool).to(
        #     t.device)  # for PyTorch versions higher than 1.2.0, use BoolTensor instead of ByteTensor
        # edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        # edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        # edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        # edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])

        edge1 = torch.zeros_like(t).to(t.device)
        edge1 = self.get_edge1(edge1, t, pad=(1, 0, 0, 0))
        edge1 = self.get_edge1(edge1, t, pad=(0, 1, 0, 0))
        edge1 = self.get_edge2(edge1, t, pad=(0, 0, 1, 0))
        edge1 = self.get_edge2(edge1, t, pad=(0, 0, 0, 1))
        return edge1.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        eps = self.eps.to(mu.device)
        return eps.mul(std) + mu
        # return mu
