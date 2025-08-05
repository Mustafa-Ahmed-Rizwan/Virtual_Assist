import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from gradio_client import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
hf_api_token = os.getenv("HF_API_TOKEN")
if not hf_api_token:
    raise ValueError("HF_API_TOKEN not found in .env file")

async def generate_image(prompt: str, model_space: str = "stabilityai/stable-diffusion") -> Dict[str, Any]:
    """
    Generate an image using the Gradio Client API for a Hugging Face Space.
    
    Args:
        prompt (str): The text prompt for image generation.
        model_space (str): The Hugging Face Space identifier (default: stabilityai/stable-diffusion).
    
    Returns:
        Dict containing the image URL or error message.
    """
    try:
        logger.info(f"Generating image for prompt: {prompt}")
        
        # Initialize Gradio Client with the Space
        client = Client(model_space)
        if hf_api_token:
            client.token = hf_api_token  # Use token for private Spaces if required

        # Call the API (using /infer as an example; adjust based on needs)
        result = client.predict(
            prompt=prompt,
            negative="",  # Optional negative prompt, leave empty if not needed
            scale=9,      # Guidance scale as per default
            api_name="/infer"
        )

        # Process the result (expecting a list of dicts with image filepath)
        if result and isinstance(result, list) and len(result) > 0 and "image" in result[0]:
            return {"image_url": result[0]["image"], "error": None}
        else:
            logger.error("Invalid response from Gradio API")
            return {"image_url": None, "error": "Failed to generate image: Invalid response from Gradio API"}

    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        return {"image_url": None, "error": f"Image generation failed: {str(e)}"}