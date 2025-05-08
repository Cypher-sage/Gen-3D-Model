# !git clone https://github.com/openai/shap-e

# # %cd shap-e

# !pip install trimesh

# pip install -e .

# !pip install rembg
# !pip install onnxruntime

# !pip install onnxruntime

# use this if using colab
# from google. colab.patches import cv2_imshow
# from google.colab import drive
from rembg import remove
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)

diffusion = diffusion_from_config(load_config('diffusion'))

def image_to3d(image_path,output_path):
  # from google. colab.patches import cv2_imshow
  # from google.colab import drive
  model = load_model('image300M', device=device)
  # drive.mount('/content/drive')



  img = load_image(image_path)
  # Output path
  output_path = output_path+('output_image.png')
  # Remove background
  output_image = remove(img)

  output_image.save(output_path)

  batch_size = 1
  guidance_scale = 3.0
  from google. colab.patches import cv2_imshow
  from google.colab import drive
  # # Mount Google Drive to access your files
  # drive.mount('/content/drive')

  # To get the best result, you should remove the background and show only the object of interest to the model.
  image = load_image(output_path)

  latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(images=[image] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,)
  render_mode = 'stf' # you can change this to 'stf' for mesh rendering
  size = 64 # this is the size of the renders; higher values take longer to render.
  cameras = create_pan_cameras(size, device)
  for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))
  for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'image_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'image_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)

# image_to3d()

def text_to_3d(prompt):
  batch_size = 1
  guidance_scale = 15.0

  model = load_model('text300M', device=device)
  latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)
  render_mode = 'nerf' # you can change this to 'stf'
  size = 64 # this is the size of the renders; higher values take longer to render.

  cameras = create_pan_cameras(size, device)
  for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))
    from shap_e.util.notebooks import decode_latent_mesh

  for i, latent in enumerate(latents):
      t = decode_latent_mesh(xm, latent).tri_mesh()
      with open(f'text_mesh_{i}.ply', 'wb') as f:
          t.write_ply(f)
      with open(f'text_mesh_{i}.obj', 'w') as f:
          t.write_obj(f)

# text_to_3d()

import trimesh


if __name__ == "__main__":
    choice = input("Please enter 'text' to proceed with text-to-3D conversion, or enter any other input to proceed with image-to-3D conversion: ")

    if choice.lower() == "text":
        prompt = input("Please enter the text prompt: ")
        text_to_3d(prompt)
    else:
        image_path = input("Please enter the image file path: ")
        print("Kindly specify the directory where the 3D output should be saved.")
        output_path = input("Please enter the destination path for saving the output: ")
        image_to_3d(image_path, output_path)
    visual=input("specify the current directory for visualization: ")
    mesh = trimesh.load(visual)
    mesh.show()
    