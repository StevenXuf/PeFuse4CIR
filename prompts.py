from utils import convert_pil_to_base64

######################################
# Prompts for FASHIONIQ dataset
######################################

def generate_composed_description_for_fashioniq(reference_image, caption):
    target_description = [
        {
            "role": "system", 
            "content": 
            """
                You are an expert at visual perception and imagination of fashion items. 
                Given a reference image of fashion items and modification instructions, mentally apply the changes and produce an accurate and complete natural-language description of the resulting fashion items. 
                The modifications may describe direct attributes (e.g., “solid white with buttons”), comparisons (e.g., “longer sleeves,” “lighter in color”), combined attributes (e.g., “black with a red cherry pattern and deep V neckline”), or negations (e.g., “no lace design”).
                Write 1 to 3 coherent sentences in English.
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": 
                    f"""
                        Here are the modification instructions: {caption}\n\n
                        Focus on the fashion item and its attributes such as type, color, pattern, material, shape, fit, and style details.
                        Ignore people and background from the image.
                        Avoid imaginary things. 
                        Be specific and objective so that I can find targeting images based on your description solely without knowing the reference image or modification instructions.
                        Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                        Now, describe how the final fashion item looks after applying the modifications. 
                        Write in 1 to 3 coherent sentences.
                    """
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
                    Ignore people and background from the image.
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Avoid unnecessary repetition or imaginary things. 
                    Write in 1 to 3 coherent sentences so that I can find similar images based on your description without seeing the given image.
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
                    Given a reference image and modification instructions, mentally apply the modifications to the reference image and describe the resulting image in clear, complete English. 
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
                    Write 1 to 3 coherent sentences.
                    """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": 
                    f"""
                        Here are the modification instructions: {caption}.\n\n
                        Focus on the elements (like objects, people, and animals), their attributes (like color, size, shape, and quantity), spatial relations, and background.
                        Avoid imaginary details and unnecessary repetitions.
                        Be specific and objective so that I can find the targeting image from an image gallery based on your description solely without knowing the reference image or modification instructions.
                        Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                        Write in 1 to 3 coherent sentences in English.
                        Now, describe how the final image looks after applying the modifications. 
                    """
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
                    Focus on elements (objects, people, animals), attributes (color, size, shape, texture, quantity), and spatial relations. 
                    Include relevant background details such as environment, lighting, and viewpoint. 
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Avoid unnecessary repetition or imaginary things.
                    Write 1 to 3 coherent sentences so that I can find similar images based on your description without seeing the given image. 
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
                        "Write 1 to 3 complete and coherent sentences."
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
            "content":
            """
                    You are an expert at visual imagination of real-world scenes. 
                    Given a reference image and modification instructions, mentally apply the modifications and produce an accurate, detailed description of the resulting scene. 
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
                    Write 1 to 3 coherent sentences in clear English.
                """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": 
                    f"""
                        Here are the modification instructions: {caption}.\n\n
                        Focus on the objects, people, animals, attributes (color, size, shape, quantity), spatial relations, and background context. 
                        Be specific and objective. 
                        Avoid imaginary details not supported by the reference image or the modification. 
                        Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                        Write 1 to 3 complete and coherent sentences so that I can find targeting images based on your description solely without knowing the reference image or modification instructions.
                        Now, describe how the final image looks after applying these modifications.
                    """
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
                    Focus on visible objects, attributes (color, size, shape, material, quantity), and spatial relations. 
                    Include relevant background details such as setting, environment, and viewpoint. 
                    Focus on the objects or attributes in the image. 
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be sure to include all relevant details.
                    Avoid unnecessary repetition or imaginary things.
                    Write 1 to 3 coherent sentences in clear English so that I can find similar images based on your description without seeing the given image.
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
#Prompts for GeneCIS dataset
###################################
def generate_composed_description_for_genecis(reference_image, caption, dataset_name):
    if dataset_name.lower() == 'change_attribute':
        caption = f'change the attribute to {caption}'
    elif dataset_name.lower() == 'change_object':
        caption = f'change the object to {caption}'
    elif dataset_name.lower() == 'focus_attribute':
        caption = f'focus on {caption} of the object'
    elif dataset_name.lower() == 'focus_object':
        caption = f'focus on {caption} in a similar setting'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} for GeneCIS")

    target_description = [
        {
            "role": "system", 
            "content":
            """
                    You are an expert at visual imagination of real-world scenes. 
                    Given a reference image and modification instructions, you should distinguish the objects and their attributes from the reference image, abd then mentally apply the modifications to the reference image, and describe the objects and their attributes in the final image after the changes. 
                    Write 1 to 3 coherent sentences in clear English.
                """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": 
                    f"""
                        Here are the modification instructions: {caption}.\n\n
                        Avoid imaginary details not supported by the reference image or the modification. 
                        Write 1 to 3 complete and coherent sentences so that I can find targeting images based on your description solely without knowing the reference image or modification instructions.
                        Now, describe the objects and their attributes after applying the modification.
                    """
                }
            ],
        }
    ]
    return target_description

def generate_target_description_for_genecis(target_image):
    target_description = [
        {
            "role": "system", 
            "content": """
                    You are an expert at visual scene perception. 
                    Given an image, describe the scene in accurate, detailed, and complete English. 
                    Focus on visible objects, and attributes (color, size, shape, material, quantity). 
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be sure to include all relevant details.
                    Avoid unnecessary repetition or imaginary things.
                    Write 1 to 3 coherent sentences in clear English so that I can find similar images based on your description without seeing the given image.
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
    elif dataset_name.lower() == "focus_attribute" or dataset_name.lower() == "change_attribute" or dataset_name.lower() == "focus_object" or dataset_name.lower() == "change_object":
        composed_description = generate_composed_description_for_genecis(reference_image, caption, dataset_name)
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
    elif dataset_name.lower() == "focus_attribute" or dataset_name.lower() == "change_attribute" or dataset_name.lower() == "focus_object" or dataset_name.lower() == "change_object":
        target_description = generate_target_description_for_genecis(target_image)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return target_description