'''
Based on: https://github.com/DPS2022/diffusion-posterior-sampling
This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.
'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
import numpy as np
from motionblur.motionblur import Kernel
import os

from .util.resizer import Resizer
from .util.img_utils import Blurkernel, fft2_m

from skimage.transform import radon, iradon
from scipy.sparse.linalg import cg
import scipy.sparse.linalg as lg
from PIL import Image, ImageDraw, ImageFont

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
        self.mask = None
    
    def forward(self, data, **kwargs):
        if self.mask is None or self.mask.shape != data.shape:
            self.mask = self.get_mask(data)
        return data * self.mask
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

    def pseudoinverse(self, measurement, **kwargs):
        return measurement
    
    def get_mask(self, data):
        image_size = data.shape[-1]
        # create a blank image with a white background
        img = Image.new("RGB", (image_size, image_size), color="white")

        # get a drawing context for the image
        draw = ImageDraw.Draw(img)
        draw.rectangle((64, 64, 192, 192), fill=(0, 0, 0))

        # convert the image to a numpy array
        img_np = np.array(img)
        img_np = img_np.transpose(2, 0, 1)
        img_th = torch.from_numpy(img_np).to(self.device)

        mask = torch.zeros(*data.shape, device=self.device)
        mask[:, img_th > 0.5] = 1.0
        return mask

@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        data = data.to(self.device) # Sending to device
        data = self.down_sample(data)
        return data.to(self.device)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    
    def pseudoinverse(self, measurement, **kwargs):
        return self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device, fix_seed=True):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity, fix_seed=fix_seed)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        data = data.to(self.device) # Sending to device
        return self.conv(data).to(self.device)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='ct')
class CTProjectionOperator(LinearOperator):
    """
    FBP implementation of CT reconstruction
    No gradients available
    """
    def __init__(self, device, num_angles=25, max_cg_iter=5, img_size=256):
        self.device = device
        self.num_angles = num_angles
        self.angles = np.linspace(0, 180, num_angles, endpoint = False)
        self.max_cg_iter = max_cg_iter
        # for cropping
        radius = img_size // 2
        img_shape = np.array([img_size, img_size])
        coords = np.array(np.ogrid[:img_size, :img_size], dtype=object)
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        self.outside_reconstruction_circle = dist > radius ** 2
    
    def crop_img(self, img):
            return 0 * self.outside_reconstruction_circle + img * (1-self.outside_reconstruction_circle)
        
    def forward(self, data, **kwargs):
        """
        y = Ax
        data: x, [-1, 1], GPU tensor with shape [B, 1, 256, 256]
        returns: sinogram of shape [B, 1, 256, num_angles], divided by 256, GPU tensor
        """
        orig_shape = data.shape
        if orig_shape[1] == 1:
            data = data.squeeze(1)
        data = (data + 1) / 2 #[-1, 1]->[0, 1]
        data = data.cpu().numpy()
        measurement = np.zeros((data.shape[0], 256, self.num_angles))
        Afun = lambda x: radon(x, theta = self.angles, circle=True)
        for i in range(data.shape[0]):
            data[i] = self.crop_img(data[i])
            measurement[i] = Afun(data[i])/256
        return torch.from_numpy(measurement).float().to(self.device).unsqueeze(1)
    
    def transpose(self, y, **kwargs):
        """
        AT
        y: sinogram of shape [B, 1, 256, num_angles], divided by 256, GPU tensor
        returns: reconstructed images of shape [B, 1, 256, 256], within scale [-1, 1]
        """
        orig_shape = y.shape
        if orig_shape[1] == 1:
            y = y.squeeze(1)
        y = y.cpu().numpy()
        fbp_train = np.zeros((y.shape[0], 256, 256))
        Atfun = lambda y: iradon(y, self.angles, filter_name= None, circle=True)
        for i in range(y.shape[0]):
            recon = Atfun(y[i])
            recon = self.crop_img(recon)
            recon = np.clip(recon, 0, 1)
            fbp_train[i] = recon
        fbp_train = fbp_train * 2 - 1
        return torch.from_numpy(fbp_train).float().to(self.device).unsqueeze(1)

    def pseudoinverse(self, sinogram, **kwargs):
        """
        A+ (pseudoinverse)
        sinogram: sinogram of shape [B, 1, 256, num_angles], divided by 256, GPU tensor
        returns: pseudoinverse of shape [B, 1, 256, 256], within scale [-1, 1]
        WARNING: This is a nonlinear operator!
        """
        orig_shape = sinogram.shape
        sinogram = sinogram.cpu().numpy()
        fwd = lambda x: radon(x, theta=self.angles, circle=True)/256
        bwd = lambda x: iradon(x, theta=self.angles, filter_name=None, circle=True) #.clip(0, 1) * 2 - 1
        LHS = lambda x: (fwd(bwd(x.reshape(256, self.num_angles)))).flatten()
        aat_linear = lg.LinearOperator((256*self.num_angles, 256*self.num_angles), matvec=LHS)
        result = np.zeros((orig_shape[0], 256, 256))
        for i in range(orig_shape[0]):
            x = cg(aat_linear, sinogram[i].flatten(), maxiter=self.max_cg_iter)[0]
            result[i] = bwd(x.reshape(256, self.num_angles))
            result[i] = self.crop_img(result[i])
        return torch.from_numpy(result*2-1).float().to(self.device).unsqueeze(1)
    
    def recon(self, gt):
        """
        ATA
        gt: ground truth images of shape [B, 1, 256, 256], within scale [-1, 1]
        returns: FBP reconstructed images of shape [B, 1, 256, 256], within scale [-1, 1]
        """
        orig_shape = gt.shape
        if orig_shape[1] == 1:
            gt = gt.squeeze(1)
        gt = (gt+1)/2   # [-1, 1]->[0, 1]
        gt = gt.float().cpu().numpy()
        gt = self.crop_img(gt)
        Afun = lambda x: radon(x, theta = self.angles, circle=False)
        Ainvfun = lambda y: iradon(y, theta = self.angles, circle=False)
        fbp_train = np.zeros((gt.shape[0], 256, 256))
        for i in range(gt.shape[0]):
            projection = Afun(gt[i])
            recon = Ainvfun(projection)
            recon = self.crop_img(recon)
            recon = np.clip(recon, 0, 1)
            fbp_train[i] = recon
        fbp_train = fbp_train * 2 - 1
        return torch.from_numpy(fbp_train).to(self.device).unsqueeze(1)

@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = "bkse/" + opt["pretrained"]
            model_path = os.path.join(work_dir, model_path)
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(data.shape[0], 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)