import argparse, os, yaml
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from util.img_utils import Blurkernel, clear_color, generate_tilt_map, mask_generator
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from motionblur.motionblur import Kernel
from ddim_sampler import *
import shutil
import lpips
import glob
import gc
import json
from tqdm import tqdm
### None Operator
class Identity:
    def forward(self, x):
        return x
### Choose the Closest Solution
def clear_color(x: torch.Tensor) -> np.ndarray:
    if torch.is_complex(x):
        x = torch.abs(x)
    
    if x.shape[1] == 3:
        x = x.detach().cpu().squeeze().numpy()
        return normalize_np(np.transpose(x, (1,2,0)))
    elif x.shape[1] == 1:
        x = x.detach().cpu().squeeze().numpy()
        return normalize_np(x)
    else:
        raise NotImplementedError
def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= -1
    img /= 2
    img = np.clip(img, a_min=0, a_max=1)
    return img
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
def calculate_metrics(ref_img, X, loss_fn_alex, device):
    output = torch.clamp(X.detach().clone(), -1, 1)
    output_numpy = output.detach().cpu().squeeze().numpy()
    output_numpy = (output_numpy + 1) / 2
    output_numpy = np.transpose(output_numpy, (1, 2, 0))
    # calculate psnr
    tmp_psnr = peak_signal_noise_ratio(ref_img, output_numpy)
    # calculate ssim
    tmp_ssim = structural_similarity(ref_img, output_numpy, channel_axis=2, data_range=1)
    # calculate lpips
    rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    gt_img_torch = torch.from_numpy(ref_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    rec_img_torch = rec_img_torch * 2 - 1
    gt_img_torch = gt_img_torch * 2 - 1
    lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
    return tmp_psnr, tmp_ssim, lpips_alex
def ilo_batch(model, scheduler, logdir, n = 100, begin = 0, end = -1, eta=0, dataset='celeba', task_config=None, device='cuda'):
    dtype = torch.float32
    measure_config = task_config['measurement']
    task_name = measure_config['operator']['name']
    if task_name == 'nonlinear_blur':
        task_name = 'nonlinear_deblur'
    elif task_name == 'blind_blur':
        kernel_type = task_config["kernel"]
        kernel_size = task_config["kernel_size"]
        task_name = kernel_type + '_deblur'
    image_paths = sorted(glob.glob(f'./data/{dataset}/*.png'))[:n]
    total_psnrs, total_ssims, total_lpipss = [], [], []
    log = os.path.join(logdir, 'log.txt')
    record = os.path.join(logdir, "record")
    os.makedirs(record, exist_ok=True)
    imgdir = os.path.join(logdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    noise_level = 0.01
    if end == -1:
        end = n
    image_paths = image_paths[begin:end]
    with open(log, "a") as f:
        print(f'DMILO: {len(image_paths)} Pictures', file = f)
        print(f'Noise Level: {noise_level}', file = f)
        if task_name in ['gaussian_deblur', 'motion_deblur']:
            print(f'Kernel Size: {kernel_size}', file = f)
    record = os.path.join(logdir, "record")
    os.makedirs(record, exist_ok=True)
    imgdir = os.path.join(logdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    ####### Hyperparameter and Time Steps
    if task_name == 'inpainting':
        lr_z, lr_nu, lr_x = 2e-2, 1e-3, 5e-1
        T = range(5)
        T_IN = range(200)
    elif task_name == 'super_resolution':
        lr_z, lr_nu, lr_x = 1e-2, 1e-3, 8
        T = range(5)
        T_IN = range(1000)
    elif task_name == 'gaussian_deblur' or task_name == 'motion_deblur':
        lr_z, lr_nu, lr_x = 8e-3, 1e-3, 3e-1
        T = range(10)
        T_IN = range(300)
    elif task_name == 'nonlinear_deblur':
        lr_z, lr_nu, lr_x = 1e-2, 1e-3, 3e-1
        T = range(10)
        T_IN = range(200)
    sample_step = len(scheduler.timesteps)
    with open(log, "a") as f:
        print(f'z\' lr = {lr_z}, nu\' lr = {lr_nu}, x\' lr = {lr_x}', file=f)    
        print(f'Outer time steps: {len(T)}, Inner time steps: {len(T_IN)}, sample step = {sample_step}', file=f)  
    ####### Load LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    for img_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        torch_seed(123)
        ### record tmp information
        max_psnr = None
        psnrs, ssims, lpipss = [], [], []
        model.eval()
        dtype = torch.float32
        gt_img_path = img_path
        gt_img = Image.open(gt_img_path).convert("RGB")
        ref_numpy = np.array(gt_img) / 255.0
        x = ref_numpy * 2 - 1
        x = x.transpose(2, 0, 1)
        ref_img = torch.Tensor(x).to(dtype).to(device).unsqueeze(0)
        ref_img.requires_grad = False
        # Prepare Operator and noise
        measure_config = task_config['measurement']
        operator = get_operator(device=device, **measure_config['operator'])
        measure_config['noise']['sigma'] = noise_level
        noiser = get_noise(**measure_config['noise'])
        if task_name == 'inpainting':
            mask_gen = mask_generator(
                **measure_config['mask_opt']
            )
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
        elif task_name in ['gaussian_deblur', 'motion_deblur']:
            intensity = task_config["intensity"]
            if kernel_type == 'motion':
                kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity).kernelMatrix
                kernel = torch.from_numpy(kernel).type(torch.float32)
                kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
            elif kernel_type == 'gaussian':
                conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device)
                kernel = conv.get_kernel().type(torch.float32)
                kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
            y = operator.forward(ref_img, kernel)
        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
        y_n = noiser(y)
        y_n.requires_grad = False
        picture_name = os.path.basename(img_path)
        tmp_imgdir = os.path.join(imgdir, os.path.splitext(picture_name)[0])
        os.makedirs(tmp_imgdir, exist_ok=True)
        plt.imsave(os.path.join(tmp_imgdir, 'measurement.png'), clear_color(y_n.clone()))
        plt.imsave(os.path.join(tmp_imgdir, 'origin.png'), clear_color(ref_img.clone()))
        img_size = img_model_config['image_size']
        X = torch.zeros((1, 3, img_size, img_size), device=device, dtype=dtype, requires_grad=True)
        plt.imsave(os.path.join(tmp_imgdir, 'init.png'), clear_color(X))
        # ILO Preparation
        target_latents = [torch.randn((1, 3, img_size, img_size), device = device, dtype = dtype, requires_grad = False) for _ in range(sample_step + 1)]
        with torch.no_grad():
            for i, tt in enumerate(scheduler.timesteps):
                t = (torch.ones(1) * tt).to(device)
                if i == 0:
                    noise_pred = model(target_latents[0], t)
                else:
                    noise_pred = model(x_t, t)
                noise_pred = noise_pred[:, :3]
                if i == 0:
                    x_t = scheduler.step(noise_pred, tt, target_latents[0], return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                else:
                    x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                target_latents[i + 1] = x_t.detach().clone().requires_grad_(False)   
            nus = [torch.zeros((1, 3, img_size, img_size), device = device, dtype = dtype, requires_grad = False) for _ in range(sample_step)]
        # End Preparation
        ##### Begin Out Loop
        for t_out in tqdm(T):
            ##### Update X
            model.eval()
            ##### Begin ILO
            # ILO preparation
            target_latents[sample_step] = y_n
            # End Preparation
            for idx in range(sample_step)[::-1]:
                # G(z_{n} + nu_{n}) -> z_{n+1}
                ###
                if idx == sample_step - 1:
                    tmp_op = operator
                    if task_name == 'inpainting' or task_name == 'boxing':
                        tmp_mask = mask
                        tmp_kernel = None
                    elif task_name in ['gaussian_deblur', 'motion_deblur']:
                        tmp_mask = None
                        tmp_kernel = kernel
                    else:
                        tmp_mask = None
                        tmp_kernel = None
                else:
                    tmp_op = Identity()
                    tmp_mask = None
                    tmp_kernel = None
                ###
                tt = scheduler.timesteps[idx]
                t = (torch.ones(1) * tt).to(device)
                # Choose the target of current layer
                target = target_latents[idx + 1].detach().clone().requires_grad_(False)
                z = target_latents[idx].detach().clone().requires_grad_(True)
                nu = nus[idx].detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([{'params': nu, 'lr': lr_nu}, {'params': z, 'lr': lr_z}])
                in_bar = tqdm(T_IN)
                ##### Inner Loop
                for _ in in_bar:
                    noise_pred = model(z, t)
                    noise_pred = noise_pred[:, :3]
                    z_hat = scheduler.step(noise_pred, tt, z, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                    z_hat = z_hat + nu
                    if tmp_mask is None and tmp_kernel is None:
                        z_hat = tmp_op.forward(z_hat)
                    elif tmp_mask is not None:
                        z_hat = tmp_op.forward(z_hat, mask=tmp_mask)
                    elif tmp_kernel is not None:
                        z_hat = tmp_op.forward(z_hat, kernel=tmp_kernel)
                    reconstruct_loss = torch.mean((z_hat - target) ** 2)
                    nu_loss = torch.mean(torch.abs(nu))
                    zprior_loss = torch.mean(z ** 2)
                    loss = reconstruct_loss + nu_loss * 1e-1 + zprior_loss * 1e-4
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    in_bar.set_postfix({'idx': idx, 'inner loss': loss.item(), 'nu loss': nu_loss.item()})
                ##### End Inner Loop
                target_latents[idx] = z.detach().requires_grad_(False)
                nus[idx] = nu.detach().requires_grad_(False)
            ##### Update Current X
            with torch.no_grad():
                for i, tt in enumerate(scheduler.timesteps):
                    t = (torch.ones(1) * tt).to(device)
                    if i == 0:
                        noise_pred = model(target_latents[0], t)
                    else:
                        noise_pred = model(x_t, t)
                    noise_pred = noise_pred[:, :3]
                    if i == 0:
                        x_t = scheduler.step(noise_pred, tt, target_latents[0], return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                    else:
                        x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
                    target_latents[i + 1] = x_t.detach().clone().requires_grad_(False) + nus[i].detach().clone().requires_grad_(False)
                    x_t = target_latents[i + 1].detach().clone().requires_grad_(False)
            ##### End Update
            target_latents[sample_step] = torch.clamp(target_latents[sample_step], -1, 1)
            X = target_latents[sample_step].detach().clone().requires_grad_(True)
            ##### End ILO
            ##### Test Current X
            with torch.no_grad():
                tar_X = X.detach().clone().requires_grad_(False)
                plt.imsave(os.path.join(tmp_imgdir, f'{t_out}_after.png'), clear_color(tar_X))
            ##### End Test Current X
            ##### End Out Loop
            #### Record Information
            with torch.no_grad():
                output = torch.clamp(X.detach().clone(), -1, 1)
                tmp_psnr, tmp_ssim, lpips_alex = calculate_metrics(ref_numpy, X, loss_fn_alex, device)
                psnrs.append(tmp_psnr)
                ssims.append(tmp_ssim)
                lpipss.append(lpips_alex)
                ###
                if max_psnr is None:
                    max_psnr = tmp_psnr
                    plt.imsave(os.path.join(tmp_imgdir, 'result.png'), clear_color(output.clone()))
                if tmp_psnr > max_psnr:
                    max_psnr = tmp_psnr
                    plt.imsave(os.path.join(tmp_imgdir, 'result.png'), clear_color(output.clone()))
        max_idx = np.argmax(psnrs)
        tmp_psnr = psnrs[max_idx]      
        tmp_ssim = ssims[max_idx]        
        lpips_alex = lpipss[max_idx]
        with open(log, "a") as f:
            picture_name = os.path.basename(img_path)
            print(f'{picture_name}: psnr: {tmp_psnr}, ssims: {tmp_ssim}, lpips: {lpips_alex}', file = f)
        ### Save record as Json
        record_dir = os.path.join(record, os.path.splitext(picture_name)[0] + '.json')
        with open(record_dir, "w") as f:
            json.dump({
                "psnr": psnrs,
                "ssim": ssims,
                "lpips_alex": lpipss
            }, f)
        ###
        total_psnrs.append(tmp_psnr)
        total_ssims.append(tmp_ssim)
        total_lpipss.append(lpips_alex)
    # Calculate avg and std
    avg_psnr = np.mean(total_psnrs)
    avg_ssim = np.mean(total_ssims)
    avg_lpips = np.mean(total_lpipss)
    with open(log, "a") as f:
        picture_name = os.path.basename(img_path)
        print(f'avg psnr: {avg_psnr}, avg ssims: {avg_ssim}, avg lpips: {avg_lpips}', file = f)
    std_psnr = np.std(total_psnrs)
    std_ssim = np.std(total_ssims)
    std_lpips = np.std(total_lpipss)
    with open(log, "a") as f:
        picture_name = os.path.basename(img_path)
        print(f'std psnr: {std_psnr}, std ssims: {std_ssim}, std lpips: {std_lpips}', file = f)
    return avg_psnr, avg_ssim, avg_lpips

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="logdir",
        default="./dmilo"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="celeba"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast sampling",
        default=3
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        nargs="?",
        help="super_resolution,inpainting,nonlinear_deblur,gaussian_deblur,motion_deblur",
        default='super_resolution'
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        nargs="?",
        help="number of test images",
        default=10
    )

    parser.add_argument(
        "-bn",
        "--begin_number",
        type=int,
        nargs="?",
        help="begin number",
        default=0
    )

    parser.add_argument(
        "-en",
        "--end_number",
        type=int,
        nargs="?",
        help="end number",
        default=-1
    )
    parser.add_argument(
        "--cuda",
        type=int,
        nargs="?",
        help="cuda device ID",
        default=1
    )
    return parser

def torch_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    # Load configurations
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    img_model_config = 'configs/model_config_{}.yaml'.format(opt.dataset)
    device = torch.device(f"cuda:{opt.cuda}")
    task_config = 'configs/tasks/{}_config.yaml'.format(opt.task)
    img_model_config = load_yaml(img_model_config)
    model = create_model(**img_model_config)
    model = model.to(device)
    model.eval()
    task_config = load_yaml(task_config)
    # Define the DDIM scheduler
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(opt.custom_steps)
    scheduler.timesteps += scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if scheduler.timesteps[0] == 1000:
        scheduler.timesteps[0] -= 1
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Put timeStep into the Log directory
    logdir = os.path.join(opt.logdir, opt.dataset, opt.task, timestamp)
    os.makedirs(logdir,exist_ok=True)
    # Intermediate Layer Optimization
    ilo_batch(model, scheduler, logdir, eta=opt.eta, dataset=opt.dataset, n=opt.number, task_config = task_config, device=device, begin=opt.begin_number, end=opt.end_number)