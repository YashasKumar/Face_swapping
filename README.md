# Face Swap using Stable Diffusion Inpainting

This project implements a face swap technique using a Stable Diffusion Inpainting model. The workflow includes loading images, detecting faces, creating masks, and performing inpainting to blend the detected face from a source image into a target image.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Functions](#functions)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

The project swaps a detected face from a source image onto a target image using the Stable Diffusion Inpainting model. It uses OpenCV for face detection and mask creation, and the `diffusers` library for inpainting.

## Dependencies

- Python 3.7+
- PyTorch
- diffusers
- OpenCV
- Pillow (PIL)
- numpy

Install dependencies with:

```bash
pip install torch diffusers opencv-python-headless pillow numpy
```

## File Structure

```
Face_Swap/
│
├── face_swap.py               # Main script
├── image1.jpg                 # Target image
├── image2.jpg                 # Source image
└── output.jpg           # Output image
```

## Functions

- **`load_model()`**: Loads the pre-trained Stable Diffusion Inpainting model and moves it to GPU if available.
  
- **`load_image(image_path)`**: Loads an image from the specified file path using OpenCV.

- **`detect_face_and_create_mask(image)`**: Detects faces in the input image and creates a mask of the detected face region.

- **`convert_to_pil(image)`**: Converts an OpenCV image to a PIL image for compatibility with the inpainting model.

- **`inpaint_face(pipe, prompt, target_image, mask_image)`**: Uses the Stable Diffusion Inpainting model to blend the source face into the target image.

- **`save_image(image, path)`**: Saves the inpainted image to a specified path and optionally displays it.

- **`main()`**:
  1. Loads the Stable Diffusion model.
  2. Loads the source and target images.
  3. Detects the face in the source image and creates a corresponding mask.
  4. Converts the target image and mask to PIL format.
  5. Inpaints the face onto the target image.
  6. Saves and displays the resulting image.

## Usage

Run the script to perform the face swap:

```bash
python face_swap.py
```

Make sure the paths to the source and target images are correctly specified in the script. The resulting image will be saved as `output_image.png` in the project directory.

## Acknowledgements

- This project uses the [Stable Diffusion Inpainting model](https://huggingface.co/runwayml/stable-diffusion-inpainting) from RunwayML.
- Face detection is performed using OpenCV's Haar Cascade classifier.

---
