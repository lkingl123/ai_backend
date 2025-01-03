import torch
import PIL
import requests
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Initialize the pipeline
def initialize_pipeline():
    model_id = "timbrooks/instruct-pix2pix"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        safety_checker=None,
        low_cpu_mem_usage=True  # Optimize memory usage
    )

    if torch.cuda.is_available():
        pipe.to("cuda")
        print(f"Using CUDA for inference on device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Falling back to CPU.")

    # Set scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

# Download an image from a URL
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Process image using the pipeline
def process_with_pipeline(pipe, image, prompt, num_steps=10, guidance_scale=1):
    print(f"Processing image with prompt: {prompt}")
    print(f"Num inference steps: {num_steps}, Guidance scale: {guidance_scale}")
    try:
        result = pipe(prompt, image=image, num_inference_steps=num_steps, image_guidance_scale=guidance_scale).images[0]
        print("Processing complete.")
        return result
    except Exception as e:
        print(f"Error during processing: {e}")
        return None
