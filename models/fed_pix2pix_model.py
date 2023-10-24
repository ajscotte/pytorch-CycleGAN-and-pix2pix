import torch
import syft as sy
from .base_model import BaseModel
from . import networks



class FedPix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            
        #defining virtual workers
        hook = sy.frameworks.torch.hook
        hook(torch) 
        self.worker1 = sy.VirtualWorker(hook, id="intel")
        
        # define networks (both generator and discriminator)
        
        #maybe need to make these models virtual
        
        
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        
        #making data into ptrs
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_ptr = (input['A' if AtoB else 'B'].to(self.device)).send(self.worker1)
        self.real_B_ptr = (input['B' if AtoB else 'A'].to(self.device)).send(self.worker1)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        
        
        
        #probably need to send to location afterwards
        #if dataloader doesn't work please use original and convert to pointers here

    def forward(self, real_A):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        fake_B = self.netG(real_A)  # G(A)
        
        return fake_B

    def backward_D(self, fake_B_ptr, model_D):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #maybe get rid of detach
        fake_AB_ptr = (torch.cat((self.real_A_ptr, fake_B_ptr), 1).detach()).send(self.worker1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        
        #send to D
        model_D = model_D(fake_AB_ptr.location)
        
        pred_fake_ptr = model_D(fake_AB_ptr)
        # pred_fake = self.netD(fake_AB_ptr.detach())
        #send to G
        self.loss_D_fake = self.criterionGAN(pred_fake_ptr, False)
        # Real
        real_AB_ptr = torch.cat((self.real_A_ptr, self.real_B_ptr), 1).send(self.worker1)
        pred_real_ptr = self.netD(real_AB_ptr)
        self.loss_D_real = self.criterionGAN(pred_real_ptr, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        
        #maybe get rid of this
        # model_D.get()

    def backward_G(self, fake_B_ptr, model_D):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        
                #maybe get rid of detach
        fake_AB_ptr = (torch.cat((self.real_A_ptr, fake_B_ptr), 1).detach()).send(self.worker1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        #send to D
        model_D = model_D(fake_AB_ptr.location)
        pred_fake_ptr = model_D(fake_AB_ptr)
        
        #start from here its bad!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.loss_G_GAN = self.criterionGAN(pred_fake_ptr, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake_B_ptr, self.real_B_ptr) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        fake_B = self.forward(self.real_A_ptr)                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D(fake_B, self.netD)                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(fake_B, self.netD)                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
