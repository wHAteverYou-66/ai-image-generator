# AI-Powered Image Generator

## 📌 Project Overview
This project is a local, web-based application that converts text descriptions into high-quality images using the open-source **Stable Diffusion v1.5** model. It provides a user-friendly interface to adjust generation parameters, save outputs, and manage image metadata.

## 🏗 Architecture
- **Model:** Stable Diffusion v1.5 (via Hugging Face Diffusers)
- **Backend:** PyTorch (Deep Learning Framework)
- **Frontend:** Streamlit (Web Interface)
- **Processing:** Local GPU execution with CPU fallback.

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10 or 3.11
- 4GB+ RAM (8GB+ recommended)

### Installation Steps
1. Clone the repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

   🖥 Hardware Requirements
GPU (Recommended): NVIDIA GPU with CUDA support. Generation time: ~5-10 seconds per image.

CPU (Fallback): Compatible with standard CPUs. Generation time: ~2-5 minutes per image.

🚀 Usage Instructions
Enter Prompt: Type a description (e.g., "A cyberpunk detective standing in the rain").

Adjust Settings: - Guidance Scale: Controls how strictly the AI follows the prompt.

Steps: Higher steps = better quality but slower speed.

Generate: Click the "Generate" button.

Save: Images are automatically saved to the generated_images folder with JSON metadata.

🛠 Technology Stack
Language: Python

Libraries: Streamlit, Diffusers, Transformers, PyTorch, Accelerate, Pillow

🎨 Prompt Engineering Tips
To get the best results, use this formula: [Subject] + [Environment] + [Art Style] + [Quality Boosters]

Example: "A futuristic city (Subject) at sunset (Environment), concept art style (Style), 4k, highly detailed, cinematic lighting (Boosters)."

⚠️ Limitations & Future Improvements
Limitations: Performance depends heavily on hardware. CPU generation is slow.

Future Improvements: 
- Add support for different aspect ratios.
- Implement "Image-to-Image" generation.
- Add user login/authentication.
