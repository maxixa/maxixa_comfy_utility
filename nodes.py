import os
import random
import re
import torch
import numpy as np
from PIL import Image
import folder_paths
import math
import torch.nn.functional as F
import sys


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

    RETURN_TYPES = ("IMAGEWildcardPromptrdPromptSK", "STRING")
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



# ComfyUI specific imports
sys.path.insert(0, '../../')
import comfy

# --- Helper Classes for Advanced Procedural Noise ---

class NoiseEngine:
    """
    Collection of pure PyTorch noise generation algorithms for GPU acceleration.
    """
    def __init__(self, device):
        self.device = device

    def _fade(self, t):
        # 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t):
        return a + t * (b - a)

    def _hash(self, x, y, seed):
        # Simple hash function for coordinates
        # Returns a pseudo-random number for x,y
        h = seed
        h += x * 374761393
        h += y * 668265263
        h = (h ^ (h >> 13)) * 1274126177
        return h

    def _gradient(self, hash_val, x, y):
        # Convert hash to gradient vector direction
        # 4 directions: (1,1), (-1,1), (1,-1), (-1,-1)
        h = hash_val & 3
        u = torch.where(h < 2, x, y)
        v = torch.where(h < 2, y, x)
        u = torch.where((h & 1) == 0, u, -u)
        v = torch.where((h & 2) == 0, v, -v)
        return u, v

    def perlin(self, shape, scale, octaves, persistence, seed):
        B, C, H, W = shape
        
        # Coordinate grid
        y_coords = torch.linspace(0, H-1, H, device=self.device)
        x_coords = torch.linspace(0, W-1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        total_noise = torch.zeros((H, W), device=self.device)
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0
        
        current_seed = seed
        
        for _ in range(octaves):
            # Scale coordinates
            f_x = grid_x * frequency * scale
            f_y = grid_y * frequency * scale
            
            # Integer coordinates
            x0 = f_x.floor().long()
            y0 = f_y.floor().long()
            
            # Fractional part
            xf = f_x - x0.float()
            yf = f_y - y0.float()
            
            # Fade curves
            u = self._fade(xf)
            v = self._fade(yf)
            
            # Wrap coordinates for hashing (using 256 limit)
            xi = x0 & 255
            yi = y0 & 255
            
            # Hash corners
            # We use a deterministic hash based on coord and seed
            aa = self._hash(xi, yi, current_seed)
            ab = self._hash(xi, yi + 1, current_seed)
            ba = self._hash(xi + 1, yi, current_seed)
            bb = self._hash(xi + 1, yi + 1, current_seed)
            
            # Gradients
            g_aa_x, g_aa_y = self._gradient(aa, xf, yf)
            g_ab_x, g_ab_y = self._gradient(ab, xf, yf - 1)
            g_ba_x, g_ba_y = self._gradient(ba, xf - 1, yf)
            g_bb_x, g_bb_y = self._gradient(bb, xf - 1, yf - 1)
            
            # Dots
            n00 = g_aa_x + g_aa_y
            n01 = g_ab_x + g_ab_y
            n10 = g_ba_x + g_ba_y
            n11 = g_bb_x + g_bb_y
            
            # Interpolate
            nx0 = self._lerp(n00, n10, u)
            nx1 = self._lerp(n01, n11, u)
            value = self._lerp(nx0, nx1, v)
            
            total_noise += value * amplitude
            max_amplitude += amplitude
            
            amplitude *= persistence
            frequency *= 2.0
            current_seed += 1 # Vary seed for each octave
            
        total_noise /= max_amplitude
        return total_noise.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    def simplex(self, shape, scale, octaves, persistence, seed):
        # Simplified approximation of Simplex noise using skewed grid logic
        # For performance in pure Torch, we approximate with warped perlin
        # True Simplex is complex to vectorize in pure Torch without compile issues
        return self.perlin(shape, scale * 1.5, octaves, persistence, seed + 100)

    def worley(self, shape, scale, octaves, persistence, seed, invert=False):
        B, C, H, W = shape
        
        # Cellular/Worley noise
        # 1. Generate random points
        # 2. Calculate distance to closest point
        
        density = int(1.0 / scale) if scale > 0 else 10
        if density < 2: density = 2
        if density > 50: density = 50 # Cap for performance
        
        grid_size = density
        
        # Create a grid of random points
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        
        # We create points in a padded grid to handle edges
        # points shape: (grid_size+2, grid_size+2, 2)
        points = torch.rand(grid_size+2, grid_size+2, 2, generator=gen, device=self.device)
        
        # Add offset for cell indices
        cell_indices = torch.arange(grid_size+2, device=self.device).float()
        row_indices = cell_indices.view(-1, 1, 1).expand(grid_size+2, grid_size+2, 1)
        col_indices = cell_indices.view(1, -1, 1).expand(grid_size+2, grid_size+2, 1)
        
        # Absolute positions of feature points
        abs_points_x = (col_indices + points[:,:,0]) * (W / grid_size)
        abs_points_y = (row_indices + points[:,:,1]) * (H / grid_size)
        
        # Create coordinate grid for every pixel
        yy, xx = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        
        # Scale factor: which cell are we in?
        cell_x = (xx.float() / (W / grid_size)).long() + 1 # +1 for padding offset
        cell_y = (yy.float() / (H / grid_size)).long() + 1
        
        # We need to check the 3x3 neighborhood of cells
        # This is expensive to do naively. 
        # Approximation: Vectorized distance check against neighbor cells.
        WildcardPrompt    # For simplicity and performance in this node:
        # We generate a dense random texture and apply a minimum filter approximation
        # OR do a proper vectorized distance check.
        
        # Proper Vectorized Worley:
        dist = torch.full((H, W), float('inf'), device=self.device)
        
        # Iterate over 3x3 neighborhood offsets (-1, 0, 1)
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Target point indices
                t_x_idx = (cell_x + i).clamp(0, grid_size+1)
                t_y_idx = (cell_y + j).clamp(0, grid_size+1)
                
                # Get point coordinates for those indices
                px = abs_points_x[t_y_idx, t_x_idx]
                py = abs_points_y[t_y_idx, t_x_idx]
                
                # Distance
                d = torch.sqrt((xx.float() - px)**2 + (yy.float() - py)**2)
                
                # Min
                dist = torch.min(dist, d)
        
        # Normalize
        noise = dist / dist.max()
        if invert:
            noise = 1.0 - noise
            
        return noise.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    def ridged(self, shape, scale, octaves, persistence, seed):
        # Ridged multifractal: 1.0 - abs(noise)
        noise = self.perlin(shape, scale, octaves, persistence, seed)
        noise = 1.0 - torch.abs(noise)
        return noise

    def billow(self, shape, scale, octaves, persistence, seed):
        # Billow: abs(noise)
        noise = self.perlin(shape, scale, octaves, persistence, seed)
        return torch.abs(noise)

    def plasma(self, shape, scale, seed):
        B, C, H, W = shape
        yy, xx = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        
        # Random parameters for plasma
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        r1, r2, r3, r4 = torch.rand(4, generator=gen, device=self.device).tolist()
        
        # Sine waves interference
        v = torch.sin(xx * scale * 0.5 + r1 * 10) + \
            torch.sin(yy * scale * 0.5 + r2 * 10) + \
            torch.sin((xx + yy) * scale * 0.5 + r3 * 10) + \
            torch.sin(torch.sqrt(xx.float()**2 + yy.float()**2) * scale * 0.5 + r4 * 10)
            
        v = (v / 4.0) # Normalize roughly
        return v.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    def domain_warp(self, shape, scale, octaves, persistence, seed):
        # Domain Warping: Warping input coordinates with noise
        B, C, H, W = shape
        
        # Base coordinates
        y_coords = torch.linspace(0, H-1, H, device=self.device)
        x_coords = torch.linspace(0, W-1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Warp X
        warp_x = self.perlin((1,1,H,W), scale, octaves, persistence, seed)
        # Warp Y
        warp_y = self.perlin((1,1,H,W), scale, octaves, persistence, seed + 999)
        
        # Apply warp
        factor = 20.0 * scale # Warp strength
        warped_x = grid_x + warp_x.squeeze() * factor
        warped_y = grid_y + warp_y.squeeze() * factor
        
        # Sample final noise at warped coordinates
        # We need a simple value noise for this that handles arbitrary floats, 
        # but perlin handles integers. We just reuse the gradient logic but with interpolation.
        # For speed, we interpolate the generated warped grid.
        
        # Let's do a second perlin lookup using the warped grid
        # (Re-implementing a single perlin lookup with arbitrary coords)
        
        # For simplicity, we treat the warped coordinates as indices into a Perlin grid
        # Scale warped coords
        f_x = warped_x * scale
        f_y = warped_y * scale
        
        # Just call perlin logic again but with custom grid?
        # To avoid recursion/recode, let's use grid_sample on the noise itself (flow noise approach)
        
        base_noise = self.perlin(shape, scale, octaves, persistence, seed)
        
        # Normalize warped coords to [-1, 1] for grid_sample
        norm_x = (warped_x / W) * 2 - 1
        norm_y = (warped_y / H) * 2 - 1
        
        grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0) # (1, H, W, 2)
        
        # Sample the base noise with the warped grid
        warped_noise = F.grid_sample(base_noise, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped_noise


# --- The Custom Node ---

class AdvancedNoiseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "noise_type": (
                    [
                        "gaussian", 
                        "perlin", 
                        "simplex", 
                        "worley", 
                        "worley_inverted",
                        "ridged", 
                        "billow", 
                        "plasma", 
                        "domain_warp"
                    ],),
                "scale": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # Image conversion options
                "image_to_noise_mode": (["none", "replace", "blend", "add"],),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "generate"
    CATEGORY = "noise"

    def generate(self, width, height, batch_size, noise_type, scale, octaves, persistence, seed, image_to_noise_mode, blend_factor, image=None):
        device = comfy.model_management.get_torch_device()
        engine = NoiseEngine(device)
        
        # Determine dimensions from image if connected
        if image is not None:
            image = image.to(device)
            batch_size, height, width, _ = image.shape
        
        # Output shape (Batch, Channel, Height, Width)
        # We use 4 channels to match latent space requirements in standard ComfyUI
        channels = 4
        shape = (batch_size, channels, height, width)

        # --- 1. Generate Base Procedural Noise ---
        if noise_type == "gaussian":
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(shape, generator=generator, device=device)
            
        elif noise_type == "perlin":
            noise = engine.perlin(shape, scale, octaves, persistence, seed)
            
        elif noise_type == "simplex":
            noise = engine.simplex(shape, scale, octaves, persistence, seed)
            
        elif noise_type == "worley":
            noise = engine.worley(shape, scale, octaves, persistence, seed, invert=False)
            
        elif noise_type == "worley_inverted":
            noise = engine.worley(shape, scale, octaves, persistence, seed, invert=True)
            
        elif noise_type == "ridged":
            noise = engine.ridged(shape, scale, octaves, persistence, seed)
            
        elif noise_type == "billow":
            noise = engine.billow(shape, scale, octaves, persistence, seed)
            
        elif noise_type == "plasma":
            noise = engine.plasma(shape, scale, seed)
            
        elif noise_type == "domain_warp":
            noise = engine.domain_warp(shape, scale, octaves, persistence, seed)
            
        else:
            # Fallback
            noise = torch.randn(shape, device=device)

        # --- 2. Normalize Procedural Noise ---
        # Ensure noise has similar stats to Gaussian (Mean 0, Std 1)
        # This is crucial for KSampler compatibility
        if noise_type != "gaussian":
            noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        # --- 3. Process Image Input ---
        if image is not None and image_to_noise_mode != "none":
            # Convert image to grayscale noise format (B, C, H, W)
            # image is (B, H, W, 3)
            if image.shape[3] == 3:
                gray = 0.299 * image[:,:,:,0] + 0.587 * image[:,:,:,1] + 0.114 * image[:,:,:,2]
            else:
                gray = image[:,:,:,0]
            
            # Normalize to Mean 0, Std 1
            gray = (gray - gray.mean()) / (gray.std() + 1e-8)
            
            # Reshape to (B, 1, H, W) and repeat to 4 channels
            img_noise = gray.unsqueeze(1).repeat(1, channels, 1, 1)

            # --- 4. Combine ---
            if image_to_noise_mode == "replace":
                noise = img_noise
            elif image_to_noise_mode == "blend":
                noise = noise * (1 - blend_factor) + img_noise * blend_factor
            elif image_to_noise_mode == "add":
                noise = noise + img_noise * blend_factor
                
            # Re-normalize after combination to prevent over-saturation
            noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        # Move to CPU for ComfyUI internal handling
        return (noise.cpu(),)



# ==============================================================================
# Node Mappings
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "Save Image (WEBP)": SaveImageWEBP,
    "Load Images From Folder": LoadImagesFromFolder,
    "Wildcard Prompt": WildcardPrompt,
    "AdvancedNoiseGenerator": AdvancedNoiseGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image (WEBP)": "Save Image (WEBP)",
    "Load Images From Folder": "Load Images From Folder",
    "Wildcard Prompt": "Wildcard Prompt",
    "AdvancedNoiseGenerator": "Advanced Noise Generator (Procedural)"
}
