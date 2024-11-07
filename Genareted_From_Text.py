# Install the necessary libraries
#pip install transformers diffusers gradio pillow

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from PIL import Image
import os

# Determine the device to use: GPU (cuda) if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model and configure it to run on the appropriate device
text_to_image_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Create the directory for generated images if it doesn't exist
output_dir = "Generated Images"
os.makedirs(output_dir, exist_ok=True)

# Create a history directory to store prompts and images
history_dir = "History"
os.makedirs(history_dir, exist_ok=True)

# Function to generate and save an image from text
def generate_and_save_image(prompt, index):
    # Create directories for history
    prompt_dir = os.path.join(history_dir, f"the_prompt_{index}")
    image_dir = os.path.join(history_dir, f"the_image_{index}")
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Save the prompt to a text file
    with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    file_path = os.path.join(image_dir, "generated_image.png")  # Define the file path for saving the image
    # Generate the image from the prompt
    images = text_to_image_model(prompt).images

    # Display the generated image
    images[0].show()

    # Save the image to the specified file path
    images[0].save(file_path)

    return file_path

# Function to handle image generation
def handle_generation(prompt):
    # Get the index for the prompt/image directories
    index = len(os.listdir(history_dir)) + 1  # Calculate index based on existing directories
    file_path = generate_and_save_image(prompt, index)

    # Load the saved image
    image = Image.open(file_path)

    return image

# Gradio UI setup
with gr.Blocks() as demo:
    gr.Markdown("# Generate Image from Text")

    # Input field for the user to describe the image they want to generate
    prompt = gr.Textbox(label="Describe the image you want to generate")

    # Submit button for image generation
    submit_btn = gr.Button("Submit")

    # Image output
    image_output = gr.Image(label="Generated Image")

    # Handle submit button click
    submit_btn.click(
        fn=handle_generation,
        inputs=[prompt],
        outputs=[image_output]
    )

# Launch the Gradio interface
demo.launch()
