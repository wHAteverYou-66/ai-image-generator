import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import datetime
import json

# --- CONFIGURATION ---
SAVE_DIR = "generated_images"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Image Generator", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_pipeline():
    """
    Loads the Stable Diffusion pipeline. 
    Uses GPU (CUDA) if available, otherwise CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use float16 for GPU to save memory, float32 for CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    st.write(f"Loading Model on **{device.upper()}**... (This may take a minute first run)")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=dtype
    )
    pipe.to(device)
    
    # Safety checker is enabled by default in diffusers, 
    # but we ensure we aren't overriding it here to maintain ethical use.
    return pipe, device

try:
    pipe, device = load_pipeline()
    st.success(f"Model loaded successfully on {device}!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("Generation Settings")

guidance_scale = st.sidebar.slider(
    "Guidance Scale", 
    min_value=1.0, max_value=20.0, value=7.5, step=0.5,
    help="Higher values make the image stick closer to the text prompt but can decrease quality."
)

num_inference_steps = st.sidebar.slider(
    "Inference Steps", 
    min_value=10, max_value=100, value=30, step=5,
    help="More steps = higher quality but slower generation."
)

num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=4, value=1)

# --- MAIN INTERFACE ---
st.title("🎨 Open-Source AI Image Generator")
st.markdown("Enter a text description below to generate an image.")

prompt = st.text_area("Enter your prompt", placeholder="A futuristic city at sunset, highly detailed, 4k...")
negative_prompt = st.text_input("Negative Prompt (Optional)", placeholder="blurry, low quality, distorted...")

# --- GENERATION LOGIC ---
if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating... Please wait."):
            try:
                # Generate the image(s)
                images = pipe(
                    prompt, 
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images
                ).images
                
                # Display and Save
                cols = st.columns(num_images)
                for idx, img in enumerate(images):
                    # Generate filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_base = f"{SAVE_DIR}/img_{timestamp}_{idx}"
                    
                    # Display in UI
                    with cols[idx]:
                        st.image(img, caption=f"Result {idx+1}", use_container_width=True)
                        
                        # Save Image
                        img_path = f"{filename_base}.png"
                        img.save(img_path)
                        
                        # Save Metadata
                        metadata = {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "guidance_scale": guidance_scale,
                            "steps": num_inference_steps,
                            "timestamp": timestamp,
                            "model": MODEL_ID
                        }
                        with open(f"{filename_base}.json", "w") as f:
                            json.dump(metadata, f, indent=4)
                            
                        # Download Button
                        with open(img_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Image",
                                data=file,
                                file_name=f"generated_{timestamp}.png",
                                mime="image/png",
                                key=f"dl_{timestamp}_{idx}"
                            )
                            
                st.success(f"Generated and saved to {SAVE_DIR}/")
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                if device == "cpu":
                    st.warning("You are running on CPU. If you run out of memory, try restarting or closing other apps.")

# --- FOOTER / DISCLAIMER ---
st.markdown("---")
st.caption("Ethical Use Disclaimer: This tool uses open-source AI. Do not use this tool to generate harmful, offensive, or illegal content. Images generated are watermarked by the model's internal safety checker where applicable.")