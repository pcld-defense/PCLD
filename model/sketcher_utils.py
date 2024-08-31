import os
import numpy as np
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import tempfile
from PIL import Image
import io
import subprocess
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import matplotlib.pyplot as plt

from util.datasets import load_image


def svg_to_png(src_path, dest_path):
    x = svg2rlg(src_path)
    renderPM.drawToFile(x, dest_path, fmt="PNG")


def show_image(img):
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.show()


def sketch_image(img, device):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig_png:
        img = img.mul(255).byte()
        img = img.permute(1, 2, 0)
        img = Image.fromarray(img.cpu().detach().numpy(), 'RGB')
        img.save(tmp_orig_png.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bmp') as tmp_bmp:
            img = Image.open(tmp_orig_png.name)
            img.save(tmp_bmp.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pbm') as tmp_pbm:
                subprocess.run(['mkbitmap', '-x', '-f', '6', tmp_bmp.name, '-o', tmp_pbm.name])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as tmp_svg:
                    subprocess.run(['potrace', '--turnpolicy', 'black',  '--turdsize',  '5', '--svg', tmp_pbm.name, '-o', tmp_svg.name])
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_png:
                        x = svg2rlg(tmp_svg.name)
                        renderPM.drawToFile(x, tmp_png.name, fmt="PNG")
                        sketched_img = load_image(tmp_png.name)
                        sketched_img = torch.tensor(sketched_img, device=device)
                        # show_image(sketched_img.detach().cpu().numpy())
                        sketched_img = sketched_img.permute(2, 0, 1)
    return sketched_img


def sketch_images(images, device):
    sketches = []
    for i in range(images.shape[0]):
        img = images[i]
        img = sketch_image(img, device)
        sketches.append(img)
    sketch = torch.stack(sketches, dim=0).to(device)
    return sketch







# def install_potrace():
#     try:
#         subprocess.run(['sudo', 'apt-get', 'update'], check=True)
#         subprocess.run(['sudo', 'apt-get', 'install', '-y', 'potrace'], check=True)
#         print("Potrace installed successfully!")
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred: {e}")


# def sketch_image(img_tensor_or_array):
#     pass
    # # Check if the input is a PyTorch Tensor and convert it to a NumPy array
    # if isinstance(img_tensor_or_array, torch.Tensor):
    #     img_np = img_tensor_or_array.detach().cpu().numpy()  # Detach and move to cpu if it's not already
    # else:
    #     img_np = img_tensor_or_array
    #
    # # Convert from (C, H, W) to (H, W, C) and normalize if necessary
    # img_np = np.transpose(img_np, (1, 2, 0))
    # if img_np.max() <= 1:
    #     img_np = (img_np * 255).astype(np.uint8)
    # else:
    #     img_np = img_np.astype(np.uint8)
    #
    # # Convert NumPy array to PIL Image
    # img_pil = Image.fromarray(img_np)
    #
    # # Convert to grayscale
    # img_gray = img_pil.convert('L')
    #
    # # Create a temporary monochrome BMP file
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.bmp') as tmp_bmp:
    #     # Convert to monochrome using a threshold
    #     monochrome = img_gray.point(lambda x: 255 if x > 128 else 0, mode='1')
    #     monochrome.save(tmp_bmp.name)
    #
    #     # Create a temporary SVG file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as tmp_svg:
    #         # Run Potrace, converting BMP to SVG
    #         os.system(f"potrace --turnpolicy black --turdsize 5 --svg {tmp_bmp.name} -o {tmp_svg.name}")
    #
    #         # Convert SVG to PNG using CairoSVG and load it into PIL Image
    #         # output_png = cairosvg.svg2png(url=tmp_svg.name)
    #         output_png = cairosvg.svg2png(url=tmp_svg.name, output_width=300,
    #                                       output_height=300, background_color="white")
    #
    # # Clean up the temporary files
    # os.unlink(tmp_bmp.name)
    # os.unlink(tmp_svg.name)
    #
    # img_sketched = Image.open(io.BytesIO(output_png))
    # img_sketched = img_sketched.convert('RGB')
    # img_sketched = transforms.ToTensor()(img_sketched)
    #
    # return img_sketched

# def sketch_images(images, device):
#     # images should be a numpy array of shape (batch_size, 3, 300, 300)
#     results = []
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         results = list(executor.map(sketch_image, [images[i] for i in range(images.shape[0])]))
#     results = torch.stack(results, dim=0).to(device)
#     return results
