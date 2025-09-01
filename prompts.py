import base64
import io

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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
                Focus on the fashion item and its attributes such as type, color, pattern, material, shape, fit, and style details in the reference image that are relevant or mentioned in the modification instructions.
                Ignore people and background from the image.
                Be concise rather than verbose.
                Be specific rather than ambiguous. 
                Be objective rather than subjective. 
                Avoid unnecessary repetition or imaginary things. 
                Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                Note that I need to find targeting images based on your description solely without knowing the reference image or modification instructions.
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
                    Ignore people and background from the image.
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose.
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
                         "Write in coherent and complete English in 1 to 3 sentences so that I can find similar images based on your description without seeing the given image."
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
    # target_description = [
    #     {
    #         "role": "system", 
    #         "content": """
    #                 You are an expert at visual imagination of real-world scenes. 
    #                 Given a reference image and modification instructions, mentally apply the modifications to the reference image and describe the resulting image in clear, complete English. 
    #                 Focus on the elements (objects, people, animals) or attributes (color, size, shape, numbers), spatial relations, and background context in the reference image that are relevant or mentioned in the modification instructions.
    #                 Apply the modifications exactly as described, and ensure the final description reflects the scene after the changes.
    #                 The modifications may include:
    #                     1. Cardinality: adjusting the number of objects (e.g., “only one bird remains”).
    #                     2. Addition: adding new objects or attributes (e.g., “add a red chair in the corner”).
    #                     3. Negation: removing elements (e.g., “remove the table”).
    #                     4. Direct Addressing: ensuring specific mentioned objects are clearly included.
    #                     5. Compare & Change: replacing one attribute with another (e.g., “same sofa but in leather”).
    #                     6. Comparative Statement: relative size, quantity, or intensity changes (e.g., “a larger group of people”).
    #                     7. Conjunction Statements: multiple modifications combined (e.g., “remove the tree and add two benches”).
    #                     8. Spatial Relations & Background: modifying positions, layout, or setting (e.g., “change the background to a beach”).
    #                     9. Viewpoint: adjusting perspective or framing (e.g., “zoom out to show the whole scene”).
    #                 Be specific rather than ambiguous. 
    #                 Be objective rather than subjective.   
    #                 Be concise rather than verbose.
    #                 Avoid imaginary details not supported by the reference image or the modification.
    #                 Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
    #                 Note that I need to find targeting images based on your description solely without seeing the reference image or modification instructions.
    #                 Write 1 to 3 coherent sentences.
    #                 """
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
    #             {
    #                 "type": "text", 
    #                 "text": (
    #                     f"Here are the modification instructions: {caption}\n\n"
    #                     "Now, describe how the final scene looks after applying the modifications." 
    #                     "Write 1 to 3 coherent and complete sentences in English so that I can find targeting images based on your description without seeing the reference image or modification instructions."
    #                 )
    #             }
    #         ],
    #     }
    # ]
    target_description = [
        # {
        #     "role": "system", 
        #     "content": """
        #             You are an expert at visual imagination of real-world scenes. 
        #             Given a reference image and modification instructions, mentally apply the modifications to the reference image and describe the resulting image in clear, specific, and complete English. 
        #             Focus on the elements (objects, people, animals) or attributes (color, size, shape, numbers), spatial relations, and background context.
        #             Pay special attention to the elements and attributes mentioned in the modification instructions.
        #             Be as specific as possible when describing the resulting image so that other people can find targeting images based on your description without knowing the reference image or modification instructions.
        #             You should follow these steps: 
        #                 1. First carefully observe the reference image and read the modification instructions;
        #                 2. Then connect the attributes and elements from modification instructions with reference image;
        #                 3. Describe the reference image;
        #                 4. Change your description based on the modification instructions;
        #                 5. Cross validate your description with the reference image and modification instructions;
        #                 6. Refine your description;
        #                 7. Finally only output the modified description.
        #             Be objective and concise.
        #             Avoid unnecessary repetition or imaginary things.
        #             Write in 1 to 3 coherent sentences.
        #             """
        # },
        {
            "role": "system",
            "content": """
                You are an expert at visually imagining real-world scenes. 
                Given a reference image and modification instructions, carefully apply the modifications 
                and produce a precise English description of the resulting image. 

                Focus on:
                - Key elements (objects, people, animals) and their attributes (color, size, shape, quantity).
                - Spatial relationships and background context.
                - Exact modifications specified in the instructions.

                Guidelines:
                1. Observe the reference image and read the modification instructions.
                2. Link the instructions to relevant elements in the reference image.
                3. Describe the original scene briefly.
                4. Update the description according to the modifications.
                5. Verify that all important details from the instructions are included.
                6. Output only the final modified description.

                Requirements:
                - Write objectively in 1 to 3 coherent sentences.
                - Be specific and clear so the description can be used for image retrieval.
                - Avoid unnecessary repetition or invented details.
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
                        "Now, describe the resulting image in 1 to 3 coherent sentences in English."
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
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose.
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
                        "Write 1 to 3 complete and coherent sentences so that I can find similar images based on your description without seeing the given image."
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
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose. 
                    Avoid imaginary details not supported by the reference image or the modification. 
                    Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                    Note that I need to find targeting images based on your description solely without seeing the reference image or modification instructions.
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
                        "Write 1 to 3 coherent and complete sentences so that I can find targeting images based on your description without seeing the reference image or modification instructions."
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
                    Be specific rather than ambiguous. 
                    Be objective rather than subjective. 
                    Be concise rather than verbose.
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
                        "Write 1 to 3 complete sentences in clear English so that I can find similar images based on your description without seeing the given image."
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