import base64
import io

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_modification_text(target_image, reference_image):
    text_modification = [
        {
            "role": "system", 
            "content": (
                "You are an expert at comparing images and identifying visual differences. "
                "Given two images (first: reference, second: target), "
                "describe all the changes needed to transform the first image into the second. "
                "Be complete and specific—mention differences in objects, colors, lighting, textures, positions, sizes, and background details. "
                "Only describe every visible changes between the images. "
                "Write in clear, full, and complete sentences in English."
            )
        },
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": "data:image;base64," + convert_pil_to_base64(reference_image)
                },
                {
                    "type": "image", 
                    "image": "data:image;base64," + convert_pil_to_base64(target_image)
                },
                {
                    "type": "text", 
                    "text": (
                        "Compare the first image to the second image. "
                        "Describe exactly what has been changed, added, or removed in order to make the first image match the second."
                    )
                }
            ],
        }
    ]
    return text_modification

def generate_composed_description(reference_image, caption):
    target_description = [
        {
            "role": "system", 
            "content": (
                "You are an expert at visual imagination. "
                "Given a reference image and modification instructions, you will mentally apply the changes and then produce a accurate, detailed and complete natural-language description of what the resulting image looks like. "
                "Only describe the final modified scene. "
                "Include colors, lighting, textures, positions, objects, people, and atmosphere. "
                "Write in clear, logical, full, and complete sentences in English."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": (
                        f"Here are the modification instructions: {caption}\n\n"
                        "Now, describe how the final image looks in coherent and complete English using at least ten tokens."
                    )
                }
            ],
        }
    ]
    return target_description

def generate_target_description(target_image):
    target_description = [
        {
            "role": "system", 
            "content": (
                "You are an expert at visual perception. "
                "Given an image, you can describe the image in an accurate, detailed and complete natural-language description. "
                "Include colors, lighting, textures, positions, objects, people, and atmosphere. "
                "Write in clear, logical, full, and complete sentences in English."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(target_image)},
                {
                    "type": "text", 
                    "text": (
                        "Now, describe what you see from the given image in coherent and complete English using at least ten tokens."
                    )
                }
            ],
        }
    ]
    return target_description