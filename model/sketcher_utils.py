import os
import numpy as np
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import tempfile
from PIL import Image
import io
import subprocess

def install_potrace():
    try:
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'potrace'], check=True)
        print("Potrace installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def sketch_image(img_tensor_or_array):
    pass
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

def sketch_images(images, device):
    # images should be a numpy array of shape (batch_size, 3, 300, 300)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(sketch_image, [images[i] for i in range(images.shape[0])]))
    results = torch.stack(results, dim=0).to(device)
    return results
