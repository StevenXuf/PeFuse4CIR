import os
from PIL import Image
import humanize

from tqdm import tqdm


def find_largest_images(directory):
    """Find the largest images in a directory by pixel dimensions after RGB conversion"""
    size_stats = {
        'max_area': 0,
        'max_width': 0,
        'max_height': 0,
        'largest_images': [],
        'widest_images': [],
        'tallest_images': []
    }

    # Supported image extensions
    image_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif')
    
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            if file.lower().endswith(image_exts):
                file_path = os.path.join(root, file)
                
                try:
                    with Image.open(file_path) as img:
                        # Convert to RGB and get dimensions
                        rgb_img = img.convert('RGB')
                        width, height = rgb_img.size
                        area = width * height
                        
                        # Update max area
                        if area > size_stats['max_area']:
                            size_stats['max_area'] = area
                            size_stats['largest_images'] = [{
                                'path': file_path,
                                'width': width,
                                'height': height,
                                'area': area,
                                'memory_bytes': area * 3
                            }]
                        elif area == size_stats['max_area']:
                            size_stats['largest_images'].append({
                                'path': file_path,
                                'width': width,
                                'height': height,
                                'area': area,
                                'memory_bytes': area * 3
                            })
                            
                        # Update max width
                        if width > size_stats['max_width']:
                            size_stats['max_width'] = width
                            size_stats['widest_images'] = [{
                                'path': file_path,
                                'width': width,
                                'height': height
                            }]
                        elif width == size_stats['max_width']:
                            size_stats['widest_images'].append({
                                'path': file_path,
                                'width': width,
                                'height': height
                            })
                            
                        # Update max height
                        if height > size_stats['max_height']:
                            size_stats['max_height'] = height
                            size_stats['tallest_images'] = [{
                                'path': file_path,
                                'width': width,
                                'height': height
                            }]
                        elif height == size_stats['max_height']:
                            size_stats['tallest_images'].append({
                                'path': file_path,
                                'width': width,
                                'height': height
                            })
                            
                except (IOError, OSError, Image.DecompressionBombError) as e:
                    print(f"\033[91mSkipping {file_path}: {str(e)}\033[0m")
    
    return size_stats

def print_results(stats):
    """Print formatted results of the image analysis"""
    if not stats['largest_images']:
        print("No valid images found in the directory")
        return
        
    # Largest by area
    print("\n\033[1mLARGEST BY AREA\033[0m")
    for img in stats['largest_images']:
        print(f"• {img['path']}")
        print(f"  Size: {img['width']}x{img['height']} = "
              f"{humanize.intcomma(img['area'])} pixels")
        print(f"  Memory: {humanize.naturalsize(img['memory_bytes'])}")
    
    # Widest
    print("\n\033[1mWIDEST IMAGES\033[0m")
    for img in stats['widest_images']:
        print(f"• {img['path']}")
        print(f"  Width: {humanize.intcomma(img['width'])}px, "
              f"Height: {humanize.intcomma(img['height'])}px")
    
    # Tallest
    print("\n\033[1mTALLEST IMAGES\033[0m")
    for img in stats['tallest_images']:
        print(f"• {img['path']}")
        print(f"  Height: {humanize.intcomma(img['height'])}px, "
              f"Width: {humanize.intcomma(img['width'])}px")

if __name__ == "__main__":
    stats = find_largest_images(directory='/data/data_fxu/RefinedFashionIQ/downloaded_images')
    print_results(stats)