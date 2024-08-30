from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import cv2
import numpy as np

def load_model():
    """Load Stable Diffusion Inpainting model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to(device)
    return pipe

def load_image(image_path):
    """Load an image from a file path."""
    return cv2.imread(image_path)

def detect_face_and_create_mask(image):
    """Detect the face and create a mask using basic methods."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    
    x, y, w, h = faces[0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    face_image = image[y:y+h, x:x+w]
    
    return mask, face_image, (x, y, w, h)

def convert_to_pil(image):
    """Convert an OpenCV image to a PIL image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def inpaint_face(pipe, prompt, target_image, mask_image):
    """Perform inpainting using the Stable Diffusion model."""
    result = pipe(prompt=prompt, image=target_image, mask_image=mask_image, strength=0.75)
    return result.images[0]

def save_image(image, path):
    """Save the PIL image to a file."""
    image.save(path)
    image.show()

def main():
    # Load model
    pipe = load_model()
    
    # Load images
    source_image_path = "D:\\YASHAS\\Projects\\Face_swap\\Screenshot 2024-08-29 123524.jpg"  # Path to your source image
    target_image_path = "D:\\YASHAS\\Projects\\Face_swap\\Screenshot 2024-08-29 123457.jpg"  # Path to your target image

    source_image = load_image(source_image_path)
    target_image = load_image(target_image_path)

    # Detect face and create mask in source image
    mask, face_image, face_coords = detect_face_and_create_mask(source_image)
    if mask is None:
        raise ValueError("No face detected in the source image.")

    # Convert target image to PIL
    target_image_pil = convert_to_pil(target_image)
    mask_pil = convert_to_pil(mask)

    # Perform inpainting to swap the face
    prompt = "Blend the detected face into the target image's face like a face swap"
    result_image = inpaint_face(pipe, prompt, target_image_pil, mask_pil)

    # Save result
    save_image(result_image, "output_image.png")

if __name__ == "__main__":
    main()