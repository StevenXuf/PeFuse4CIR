import base64
import io

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_modification(target_image, reference_image):
    text_modification = [
        {
            "role": "system", 
            "content": (
                "You are an expert at comparing images and identifying visual differences. "
                "Given two images (first: reference, second: target), "
                "Describe all the changes needed to transform the first image into the second. "
                "Be complete and specific—mention differences in objects, colors, lighting, textures, positions, sizes, and background details. "
                "Only describe every visible changes between the images. "
                "Write in clear, full, and complete sentences in English."
            )
        },
        {
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

######################################
# Prompts for FASHIONIQ dataset
######################################

def generate_composed_description_for_fashioniq(reference_image, caption):
    target_description = [
        {
            "role": "system", 
            "content": 
            """
                You are an expert at visual imagination of fashion items. 
                Given a reference image of fashion items and modification instructions, mentally apply the changes and produce an accurate and complete natural-language description of the resulting fashion items. 
                The modifications may describe direct attributes (e.g., “solid white with buttons”), comparisons (e.g., “longer sleeves,” “lighter in color”), combined attributes (e.g., “black with a red cherry pattern and deep V neckline”), or negations (e.g., “no lace design”).
                Focus on the objects or attributes such as type, color, pattern, material, shape, fit, and style details in the reference image that are relevant or mentioned in the modification instructions.
                Ignore people and irrelevant attributes like background from image.
                Be concrete rather than ambiguous. 
                Be objective rather than subjective. 
                Be concise rather than verbose.
                Avoid unnecessary repetition or imaginary things. 
                Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a top-down angle (if mentioned) instead of unchanged angle, etc.
                Write 1 to 3 coherent sentences in clear English.
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": (
                        f"Here are the modification instructions: {caption}\n\n"
                        "Now, describe how the final fashion item looks after applying the modifications. "
                        "Write in coherent and complete English in 1 to 3 sentences."
                    )
                }
            ],
        }
    ]
    return target_description

def generate_target_description_for_fashioniq(target_image):
    target_description = [
        {
            "role": "system", 
            "content": """
                    You are an expert at visual perception of fashion items. 
                    Given an image, describe the fashion item in clear, accurate, and complete English. 
                    Focus on visual attributes of the clothes such as type, color, texture, pattern, material, shape, fit, and style details. 
                    Ignore people and irrelevant attributes like background from image.
                    Be concrete rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose.
                    Avoid unnecessary repetition or imaginary things. 
                    Write in 1 to 3 coherent sentences.
                """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(target_image)},
                {
                    "type": "text", 
                    "text": (
                        "Now, describe the fashion item from the given image. "
                         "Write in coherent and complete English in 1 to 3 sentences."
                    )
                }
            ],
        }
    ]
    return target_description


###################################
# Prompts for CIRR dataset
###################################
def generate_composed_description_for_cirr(reference_image, caption):
    target_description = [
        {
            "role": "system", 
            "content": """
                    You are an expert at visual imagination of real-world scenes. 
                    Given a reference image and modification instructions, mentally apply the modifications and describe the final image in clear, complete English. 
                    Focus on the elements (objects, people, animals) or attributes (color, size, shape, numbers), spatial relations, and background context in the reference image that are relevant or mentioned in the modification instructions.
                    Ignore irrelevant attributes and elements in the reference image that are not relevant or mentioned in the modification instructions. 
                    Apply the modifications exactly as described, and ensure the final description reflects the scene after the changes.
                    The modifications may include:
                        1. Cardinality: adjusting the number of objects (e.g., “only one bird remains”).
                        2. Addition: adding new objects or attributes (e.g., “add a red chair in the corner”).
                        3. Negation: removing elements (e.g., “remove the table”).
                        4. Direct Addressing: ensuring specific mentioned objects are clearly included.
                        5. Compare & Change: replacing one attribute with another (e.g., “same sofa but in leather”).
                        6. Comparative Statement: relative size, quantity, or intensity changes (e.g., “a larger group of people”).
                        7. Conjunction Statements: multiple modifications combined (e.g., “remove the tree and add two benches”).
                        8. Spatial Relations & Background: modifying positions, layout, or setting (e.g., “change the background to a beach”).
                        9. Viewpoint: adjusting perspective or framing (e.g., “zoom out to show the whole scene”).
                    Be concrete rather than ambiguous. 
                    Be objective rather than subjective.   
                    Be concise rather than verbose.
                    Avoid imaginary details not supported by the reference image or the modification.
                    Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                    Write 1 to 3 coherent sentences in clear English.
                    """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": (
                        f"Here are the modification instructions: {caption}\n\n"
                        "Now, describe how the final scene looks after applying the modifications." 
                        "Write 1 to 3 coherent and complete sentences in English."
                    )
                }
            ],
        }
    ]
    return target_description

def generate_target_description_for_cirr(target_image):
    target_description = [
        {
            "role": "system", 
            "content": """
                    You are an expert at visual scene description. 
                    Given an image, describe the scene in accurate, detailed, and complete English. 
                    Mention all important elements (objects, people, animals), attributes (color, size, shape, texture, number), and spatial relations. 
                    Focus on the elements or attributes in the image.
                    Include relevant background details such as environment, lighting, and viewpoint. 
                    Be concrete rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose.
                    Avoid unnecessary repetition or imaginary things.
                    Write 1 to 3 coherent sentences that capture everything needed to uniquely identify the image. 
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(target_image)},
                {
                    "type": "text", 
                    "text": (
                        "Now, describe the scene from the given image. "
                        "Write 1 to 3 complete and coherent sentences in English."
                    )
                }
            ],
        }
    ]
    return target_description




###################################
#Prompts for CIRCO dataset
###################################
def generate_composed_description_for_circo(reference_image, caption):
    target_description = [
        {
            "role": "system", 
            "content": """
                    You are an expert at visual imagination of real-world scenes. 
                    Given a reference image and modification instructions, mentally apply the modifications and produce an accurate, detailed description of the resulting scene. 
                    Focus on the elements (objects, people, animals) or attributes (color, size, shape, numbers), spatial relations, and background context in the reference image that are relevant or mentioned in the modification instructions.
                    Ignore irrelevant attributes and elements in the reference image that are not relevant or mentioned in the modification instructions. 
                    Apply the modifications exactly as described, and describe the final scene after the changes.
                    The modifications may involve:
                        1. Cardinality: adjusting the number of objects (e.g., “has two boxes”).
                        2. Addition: introducing new objects or attributes (e.g., “a child under the umbrella”).
                        3. Negation: removing elements (e.g., “shows no bike”).
                        4. Direct Addressing: ensuring a specific object is present (e.g., “next to a window”).
                        5. Compare & Change: altering attributes (e.g., “different color,” “surrounded by flowers”).
                        6. Comparative Statements: relative size, number, or intensity (e.g., “more stickers,” “larger crowd”).
                        7. Conjunction Statements: multiple edits at once (e.g., “surrounded by snow and trees are more bare”).
                        8. Spatial Relations & Background: positioning or environment changes (e.g., “skyscrapers in the background”).
                        9. Viewpoint: changes in perspective or framing (e.g., “shot from above”).
                    Be concrete rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose. 
                    Avoid imaginary details not supported by the reference image or the modification. 
                    Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                    Write 1 to 3 coherent sentences in clear English.
                """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": (
                        f"Here are the modification instructions: {caption}\n\n"
                        "Now, describe how the final scene looks after applying these modifications."
                        "Write 1 to 3 coherent and complete sentences."
                    )
                }
            ],
        }
    ]
    return target_description

def generate_target_description_for_circo(target_image):
    target_description = [
        {
            "role": "system", 
            "content": """
                    You are an expert at visual scene perception. 
                    Given an image, describe the scene in accurate, detailed, and complete English. 
                    Mention all visible objects, attributes (color, size, shape, material, number), and spatial relations. 
                    Include relevant background details such as setting, environment, and viewpoint. 
                    Focus on the objects or attributes in the image. 
                    Be concrete rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose.
                    Avoid unnecessary repetition or imaginary things.
                    Write 1 to 3 coherent sentences in clear English.
                    """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(target_image)},
                {
                    "type": "text", 
                    "text": (
                        "Now, describe the scene from the given image. "
                        "Write 1 to 3 complete sentences in clear English."
                    )
                }
            ],
        }
    ]
    return target_description


####################################
# Prompts redirection
####################################
def get_composed_prompts(dataset_name, reference_image, caption):
    if dataset_name.lower() == "circo":
        composed_description = generate_composed_description_for_circo(reference_image, caption)
    elif dataset_name.lower() == "cirr":
        composed_description = generate_composed_description_for_cirr(reference_image, caption)
    elif dataset_name.lower() == "fashioniq" or dataset_name.lower() == "refinedfashioniq":
        composed_description = generate_composed_description_for_fashioniq(reference_image, caption)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return composed_description


def get_target_prompts(dataset_name, target_image):
    if dataset_name.lower() == "circo":
        target_description = generate_target_description_for_circo(target_image)
    elif dataset_name.lower() == "cirr":
        target_description = generate_target_description_for_cirr(target_image)
    elif dataset_name.lower() == "fashioniq" or dataset_name.lower() == "refinedfashioniq":
        target_description = generate_target_description_for_fashioniq(target_image)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return target_description