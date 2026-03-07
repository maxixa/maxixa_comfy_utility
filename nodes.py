import os
import random
import re
import torch
import numpy as np
from PIL import Image
import folder_paths

# ==============================================================================
# NODE 1: Save Image (WEBP)
# ==============================================================================

class SaveImageWEBP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1}),
                "lossless": ("BOOLEAN", {"default": False}),
                "remove_metadata": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix, quality, lossless, remove_metadata, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            try:
                digits = int(filename[prefix_len+1:].split('_')[0])
            except:
                digits = 0
            return digits

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(self.output_dir, subfolder)
        
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        existing_files = [f for f in os.listdir(full_output_folder) if f.endswith('.webp') and f.startswith(os.path.basename(filename_prefix))]
        if existing_files:
            counters = [map_filename(f) for f in existing_files]
            counter = max(counters) + 1
        else:
            counter = 1

        results = list()
        
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            file = f"{filename_prefix}_{counter:05}_.webp"
            full_path = os.path.join(full_output_folder, file)
            
            img.save(full_path, format='WEBP', quality=quality, lossless=lossless)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}

# ==============================================================================
# NODE 2: Load Images From Folder
# ==============================================================================

_folder_counters = {}

class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "mode": (["increment", "random", "fixed_index"], {"default": "increment"}),
                "fixed_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, folder_path, mode, fixed_index, seed=0):
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(folder_paths.base_path, folder_path)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        files.sort()

        if len(files) == 0:
            raise FileNotFoundError(f"No images found in folder: {folder_path}")

        current_index = 0

        if mode == "fixed_index":
            current_index = fixed_index % len(files)
        elif mode == "random":
            random.seed(seed)
            current_index = random.randint(0, len(files) - 1)
        elif mode == "increment":
            if folder_path not in _folder_counters:
                _folder_counters[folder_path] = 0
            
            current_index = _folder_counters[folder_path]
            
            next_index = current_index + 1
            if next_index >= len(files):
                next_index = 0
            _folder_counters[folder_path] = next_index

        selected_file = files[current_index]
        full_path = os.path.join(folder_path, selected_file)

        img = Image.open(full_path)
        
        output_mask = None
        output_image = None

        if img.mode == 'RGBA':
            img_data = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(img_data[:, :, :3]).unsqueeze(0)
            output_mask = torch.from_numpy(img_data[:, :, 3]).unsqueeze(0)
        elif 'A' in img.mode:
            img = img.convert('RGBA')
            img_data = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(img_data[:, :, :3]).unsqueeze(0)
            output_mask = torch.from_numpy(img_data[:, :, 3]).unsqueeze(0)
        else:
            img = img.convert('RGB')
            img_data = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(img_data).unsqueeze(0)
            output_mask = torch.ones((1, img_data.shape[0], img_data.shape[1]), dtype=torch.float32)

        return (output_image, output_mask, selected_file)

    @classmethod
    def IS_CHANGED(s, folder_path, mode, fixed_index, seed=0):
        return float("NaN")

# ==============================================================================
# NODE 3: Wildcard Prompt Processor
# ==============================================================================

class WildcardPrompt:
    """
    Takes a text input with {option1|option2} syntax and outputs a string
    with one option randomly selected.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "{big|small} city"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_text"
    CATEGORY = "utils"

    def process_text(self, text, seed):
        # Set the seed for reproducibility
        random.seed(seed)
        
        # Regex to find curly braces { }
        # Pattern explanation:
        # \{      -> Matches opening brace
        # ([^}]+) -> Group 1: matches one or more chars that are NOT a closing brace
        # \}      -> Matches closing brace
        pattern = re.compile(r'\{([^{}]+)\}')
        
        def replace_match(match):
            content = match.group(1)
            # Split by pipe symbol |
            options = content.split('|')
            # Clean whitespace from each option
            options = [opt.strip() for opt in options]
            
            # If valid options exist, pick one
            if options:
                return random.choice(options)
            return ""

        # Replace all occurrences found by the regex
        result = pattern.sub(replace_match, text)
        
        return (result,)

# ==============================================================================
# Node Mappings
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "Save Image (WEBP)": SaveImageWEBP,
    "Load Images From Folder": LoadImagesFromFolder,
    "Wildcard Prompt": WildcardPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image (WEBP)": "Save Image (WEBP)",
    "Load Images From Folder": "Load Images From Folder",
    "Wildcard Prompt": "Wildcard Prompt"
          }
