import os
import sys
import shutil
import random
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import colorsys

# Try to import toml for Kohya SS config generation
try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    print("Warning: toml package not available. Kohya SS config files will be generated as JSON instead.")
    TOML_AVAILABLE = False

from PyQt5 import QtWidgets, QtCore
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}
FUN_MESSAGES = [
    "Crunching numbers...",
    "Sampling some greatness...",
    "Copying with style...",
]

@dataclass
class FolderEntry:
    path: str
    percent: int = 0
    blank: int = 5
    basic: int = 15
    detailed: int = 30
    structured: int = 50
    # Per-folder Kohya SS settings for LoRA/DreamBooth mode
    instance_prompt: str = ""
    class_prompt: str = "a person"  # Changed from "person" to "a person"
    repeats: int = 3

    def __hash__(self):
        """Make FolderEntry hashable so it can be used as a dictionary key"""
        return hash(self.path)
    
    def __eq__(self, other):
        """Define equality based on path for consistent hashing"""
        if isinstance(other, FolderEntry):
            return self.path == other.path
        return False

    def caption_total(self) -> int:
        return self.blank + self.basic + self.detailed + self.structured

class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, entries: List[FolderEntry], dest: str, total_count: int, mode: str, use_megapixels: bool = False, convert_to_png: bool = True, regularization_folder: str = None, regularization_percent: int = 0, training_seed: int = None, regularization_seed: int = None, selected_reg_folders: List[str] = None, kohya_project: bool = False, training_mode: str = "LoRA/DreamBooth", instance_prompt: str = "", class_prompt: str = "person", repeats: int = 3, copy_training: bool = True, copy_regularization: bool = True, resize_enabled: bool = True, resize_pixels: int = 1024):
        super().__init__()
        self.entries = entries
        self.dest = dest
        self.total_count = total_count
        self.mode = mode
        self.use_megapixels = use_megapixels
        self.convert_to_png = convert_to_png
        self.regularization_folder = regularization_folder
        self.regularization_percent = regularization_percent
        self.training_seed = training_seed
        self.regularization_seed = regularization_seed
        self.selected_reg_folders = selected_reg_folders or []
        self.kohya_project = kohya_project
        self.training_mode = training_mode
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.repeats = repeats
        self.copy_training = copy_training
        self.copy_regularization = copy_regularization
        self.resize_enabled = resize_enabled
        self.resize_pixels = resize_pixels
        self.log_file = os.path.join(dest, "sampler.log")
        
        # Create a custom logger to avoid conflicts with global logging
        self.logger = logging.getLogger(f"sampler_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        self.file_handler = logging.FileHandler(self.log_file, mode='w')
        self.file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.file_handler)

    def run(self):
        # Set training seed if provided for consistent training image selection
        if self.training_seed is not None:
            random.seed(self.training_seed)
            self.logger.info("Using training seed: %d", self.training_seed)
        
        # --- TRAINING IMAGES: Only process if copy_training is True ---
        if self.copy_training:
            total_images = sum(len(self._list_images(e.path)) for e in self.entries)
            processed = 0
            self.message.emit(random.choice(FUN_MESSAGES))
            self.logger.info("Starting sampling to %s", os.path.join(self.dest, "img"))
            tasks = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for entry in self.entries:
                    images = self._list_images(entry.path)
                    quota = int(len(images) * entry.percent / 100)  # Use folder's own image count and percent
                    if quota > len(images):
                        quota = len(images)
                    self.logger.info("%s: selecting %d images", entry.path, quota)
                    selected = random.sample(images, quota)
                    if self.training_mode == "LoRA/DreamBooth":
                        # --- LoRA/DreamBooth: Use folder name as '{repeats}_{instance_prompt} {class_prompt}' ---
                        instance_prompt = (entry.instance_prompt or self.instance_prompt).strip()
                        class_prompt = (entry.class_prompt or self.class_prompt).strip()
                        repeats = entry.repeats or self.repeats
                        if instance_prompt:
                            folder_name = f"{repeats}_{instance_prompt} {class_prompt}".strip()
                        else:
                            folder_name = f"{repeats}_{class_prompt}".strip()
                        # Remove any forbidden filesystem characters
                        folder_name = ''.join(c if c not in '\\/:*?"<>|' else '_' for c in folder_name)
                        subfolder = os.path.join(os.path.join(self.dest, "img"), folder_name)
                        os.makedirs(subfolder, exist_ok=True)
                        for img in selected:
                            base = os.path.basename(img)
                            name, ext = os.path.splitext(base)
                            dest_img = os.path.join(subfolder, base)
                            dest_caption = os.path.join(subfolder, name + ".txt")
                            tasks.append(executor.submit(self._copy_image_only, img, dest_img, entry, name, dest_caption))
                    else:
                        # --- Fine-tuning: Save all images directly in img/ (no subfolders) ---
                        for img in selected:
                            base = os.path.basename(img)
                            name, ext = os.path.splitext(base)
                            dest_img = os.path.join(os.path.join(self.dest, "img"), base)
                            dest_caption = os.path.join(os.path.join(self.dest, "img"), name + ".txt")
                            tasks.append(executor.submit(self._copy_image_only, img, dest_img, entry, name, dest_caption))
                for future in as_completed(tasks):
                    result = future.result()
                    processed += 1
                    if total_images:
                        self.progress.emit(int(processed / total_images * 100))
                    if processed % 10 == 0:
                        self.message.emit(random.choice(FUN_MESSAGES))
            # --- PNG conversion in-place inside each subfolder of img/ ---
            if self.convert_to_png:
                img_dir = os.path.join(self.dest, "img")
                self.message.emit("Converting non-PNG images to PNG in-place...")
                for subdir, dirs, files in os.walk(img_dir):
                    non_png_files = []
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext != ".png" and ext in IMAGE_EXTENSIONS:
                            non_png_files.append(os.path.join(subdir, f))
                    if non_png_files:
                        self.message.emit(f"Converting {len(non_png_files)} images to PNG in {subdir}...")
                        converted = 0
                        max_workers = min(os.cpu_count(), 16)
                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            png_tasks = []
                            for src_path in non_png_files:
                                png_path = os.path.splitext(src_path)[0] + ".png"
                                png_tasks.append(executor.submit(convert_and_delete, src_path, png_path))
                            for future in as_completed(png_tasks):
                                try:
                                    result = future.result()
                                    if result is True:
                                        converted += 1
                                    else:
                                        self.message.emit(f"PNG conversion error: {result}")
                                    png_progress = int((converted / len(non_png_files)) * 100)
                                    self.progress.emit(png_progress)
                                    if converted % 5 == 0:
                                        self.message.emit(f"Converted {converted}/{len(non_png_files)} images to PNG in {subdir}...")
                                except Exception as e:
                                    self.message.emit(f"PNG conversion error: {e}")
                        self.message.emit(f"PNG conversion complete: {converted} images converted in {subdir}")
                    else:
                        self.message.emit(f"No images need PNG conversion in {subdir}")
        else:
            self.message.emit("Copy Training Set is unchecked: training images were left untouched.")

        # --- REGULARIZATION IMAGES: Only process if copy_regularization is True ---
        if self.training_mode != "Fine-tuning" and self.copy_regularization and self.regularization_folder and self.regularization_percent > 0:
            self._process_regularization()
        elif not self.copy_regularization:
            self.message.emit("Copy Regularization Set is unchecked: regularization images were left untouched.")
        if self.kohya_project:
            self._create_kohya_structure()
        self.close_log_file()
        self.finished.emit()

    def _process_regularization(self):
        """Copy regularization images directly into the root 'regularization' folder, using a global quota distributed across selected folders."""
        try:
            if self.regularization_seed is not None:
                random.seed(self.regularization_seed)
                self.logger.info("Using regularization seed: %d", self.regularization_seed)

            self.message.emit("Processing regularization images...")

            # Create regularization folder if it doesn't exist
            reg_dest = os.path.join(self.dest, "regularization")
            os.makedirs(reg_dest, exist_ok=True)

            # Collect all selected regularization folders
            reg_folders = self.selected_reg_folders if self.selected_reg_folders else []
            if not reg_folders:
                self.message.emit("No regularization folders selected")
                return

            # Calculate the total number of regularization images to copy
            reg_count = int(self.total_count * self.regularization_percent / 100)
            if reg_count == 0:
                self.message.emit("Regularization percent or total images is zero; nothing to copy.")
                return

            # Gather all available images from each selected folder
            folder_images = []  # List of lists
            for folder_name in reg_folders:
                if folder_name == "root":
                    search_path = self.regularization_folder
                else:
                    search_path = os.path.join(self.regularization_folder, folder_name)
                images = []
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            images.append(os.path.join(root, file))
                folder_images.append(images)

            total_available = sum(len(imgs) for imgs in folder_images)
            if total_available == 0:
                self.message.emit("No regularization images found in selected folders.")
                return
            if reg_count > total_available:
                reg_count = total_available
                self.message.emit(f"Not enough regularization images; using all {total_available} available.")

            # Distribute reg_count evenly across folders
            num_folders = len(folder_images)
            images_per_folder = reg_count // num_folders
            remainder = reg_count % num_folders
            selected_reg_images = []
            used_images = set()
            for i, images in enumerate(folder_images):
                quota = images_per_folder + (1 if i < remainder else 0)
                take = min(quota, len(images))
                if take > 0:
                    chosen = random.sample(images, take)
                    selected_reg_images.extend(chosen)
                    used_images.update(chosen)

            # If we still need more, fill from all remaining images
            if len(selected_reg_images) < reg_count:
                all_remaining = [img for imgs in folder_images for img in imgs if img not in used_images]
                need = reg_count - len(selected_reg_images)
                if len(all_remaining) >= need:
                    selected_reg_images.extend(random.sample(all_remaining, need))
                else:
                    selected_reg_images.extend(all_remaining)

            reg_processed = 0
            tasks = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for img_path in selected_reg_images:
                    base = os.path.basename(img_path)
                    name, ext = os.path.splitext(base)
                    dest_img = os.path.join(reg_dest, base)
                    dest_caption = os.path.join(reg_dest, name + ".txt")
                    tasks.append(executor.submit(self._copy_regularization_image, img_path, dest_img, dest_caption))
                for future in as_completed(tasks):
                    result = future.result()
                    reg_processed += 1
                    if reg_processed % 10 == 0:
                        self.message.emit(f"Processed {reg_processed} regularization images...")
            self.message.emit(f"Regularization complete: {reg_processed} images copied")
        except Exception as e:
            self.message.emit(f"Regularization error: {e}")

    def _get_unique_dest_path(self, dest_img):
        """Return a unique destination path by appending _1, _2, etc if needed."""
        base, ext = os.path.splitext(dest_img)
        counter = 1
        unique_dest = dest_img
        while os.path.exists(unique_dest):
            unique_dest = f"{base}_{counter}{ext}"
            counter += 1
        return unique_dest

    def _copy_regularization_image(self, img_path, dest_img, dest_caption):
        """Copy a regularization image and look for its caption, ensuring unique filenames."""
        try:
            # Ensure unique image path
            dest_img = self._get_unique_dest_path(dest_img)
            # Ensure caption gets same suffix
            base_img, ext_img = os.path.splitext(dest_img)
            dest_caption = base_img + ".txt"
            shutil.copy2(img_path, dest_img)
            # Look for caption file in same directory as image
            img_dir = os.path.dirname(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            caption_file = os.path.join(img_dir, f"{img_name}.txt")
            if os.path.exists(caption_file):
                shutil.copy2(caption_file, dest_caption)
                self.logger.info("%s -> %s (with caption)", img_path, dest_img)
            else:
                open(dest_caption, "w").close()
                self.logger.info("%s -> %s (no caption)", img_path, dest_img)
            return True
        except Exception as e:
            self.message.emit(f"Failed to copy regularization image {img_path}: {e}")
            return False

    def close_log_file(self):
        """Properly close the log file to release the file handle"""
        try:
            if hasattr(self, 'file_handler'):
                self.file_handler.close()
                self.logger.removeHandler(self.file_handler)
        except Exception:
            pass  # Ignore errors during cleanup

    def _list_images(self, folder: str) -> List[str]:
        files = []
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                files.append(os.path.join(folder, f))
        return files

    def _copy_image_only(self, img, dest_img, entry, name, dest_caption):
        try:
            from PIL import Image
            # Ensure unique image path
            dest_img = self._get_unique_dest_path(dest_img)
            # Ensure caption gets same suffix
            base_img, ext_img = os.path.splitext(dest_img)
            dest_caption = base_img + ".txt"
            if self.resize_enabled:
                with Image.open(img) as im:
                    w, h = im.size
                    target = self.resize_pixels
                    if w <= 0 or h <= 0:
                        raise Exception(f"Invalid image size: {w}x{h}")
                    # Determine scale factor for shortest side
                    if w < h:
                        new_w = target
                        new_h = int(h * (target / w))
                    else:
                        new_h = target
                        new_w = int(w * (target / h))
                    im = im.resize((new_w, new_h), Image.LANCZOS)
                    im.save(dest_img)
            else:
                shutil.copy2(img, dest_img)
            style = self._copy_caption(entry, name, dest_caption, self.training_mode)
            self.logger.info("%s -> %s (%s)", img, dest_img, style)
            return True
        except Exception as e:
            self.message.emit(f"Failed to copy {img}: {e}")
            return False

    def _copy_caption(self, entry: FolderEntry, name: str, dest_caption: str, training_mode: str) -> str:
        if training_mode == "Fine-tuning":
            styles = ["blank", "basic", "detailed", "structured"]
            weights = [entry.blank, entry.basic, entry.detailed, entry.structured]
        else:  # LoRA/DreamBooth
            styles = ["basic", "detailed", "structured"]
            weights = [entry.basic, entry.detailed, entry.structured]
        style = random.choices(styles, weights=weights)[0]
        if style == "blank":
            open(dest_caption, "w").close()
            return style
        # Look for caption file in subfolder (e.g., basic/image001.txt)
        caption_file = os.path.join(entry.path, style, f"{name}.txt")
        if not os.path.exists(caption_file):
            # Fallback to traditional dataset structure (image001.txt)
            caption_file = os.path.join(entry.path, f"{name}.txt")
        if os.path.exists(caption_file):
            shutil.copy2(caption_file, dest_caption)
        else:
            open(dest_caption, "w").close()
        return style

    def _create_kohya_structure(self):
        """Create Kohya SS project structure from sampled images"""
        try:
            self.message.emit("Creating Kohya SS project structure...")
            
            # Create the Kohya SS directory structure
            img_dir = os.path.join(self.dest, "img")
            regularization_dir = os.path.join(self.dest, "regularization")
            os.makedirs(img_dir, exist_ok=True)
            
            # Identify all images in the destination folder
            dest_images = []
            for f in os.listdir(self.dest):
                if os.path.isfile(os.path.join(self.dest, f)):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        dest_images.append(f)
            
            if not dest_images and not self.copy_training:
                self.message.emit("No training images to organize")
                return
            
            training_folders = {}
            
            # Organize training images by source folder if copy_training is enabled
            if self.copy_training and dest_images:
                self.message.emit("Organizing training images into Kohya SS structure...")
                if self.training_mode == "LoRA/DreamBooth":
                    for entry in self.entries:
                        if not entry.path or not os.path.isdir(entry.path):
                            continue
                        # Use per-folder settings for Kohya SS
                        instance_prompt = entry.instance_prompt or self.instance_prompt
                        class_prompt = entry.class_prompt or self.class_prompt
                        repeats = entry.repeats or self.repeats
                        # Create folder name: {repeats}_{instance_prompt}_{class_prompt}
                        folder_name = f"{repeats}_{instance_prompt}_{class_prompt}".strip()
                        # Clean folder name of invalid characters and replace spaces with underscores
                        folder_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in folder_name)
                        folder_name = "_".join(folder_name.split())
                        training_folders[entry] = folder_name
                        training_folder = os.path.join(img_dir, folder_name)
                        os.makedirs(training_folder, exist_ok=True)
                    # Move images from source folders to their respective training folders
                    for image_file in dest_images:
                        image_path = os.path.join(self.dest, image_file)
                        caption_file = os.path.join(self.dest, os.path.splitext(image_file)[0] + ".txt")
                        moved = False
                        for entry in self.entries:
                            if not entry.path or not os.path.isdir(entry.path):
                                continue
                            source_image = os.path.join(entry.path, image_file)
                            if os.path.exists(source_image):
                                folder_name = training_folders[entry]
                                training_folder = os.path.join(img_dir, folder_name)
                                dest_image = os.path.join(training_folder, image_file)
                                shutil.move(image_path, dest_image)
                                if os.path.exists(caption_file):
                                    dest_caption = os.path.join(training_folder, os.path.splitext(image_file)[0] + ".txt")
                                    shutil.move(caption_file, dest_caption)
                                moved = True
                                break
                        # If image wasn't matched to a source folder, skip it (do NOT create a default folder)
                        if not moved:
                            # Optionally, log or warn about unmatched images
                            self.logger.info(f"Skipped unmatched image: {image_file}")
                else:  # Fine-tuning mode
                    # All images go directly into img/ (no subfolders)
                    for image_file in dest_images:
                        image_path = os.path.join(self.dest, image_file)
                        caption_file = os.path.join(self.dest, os.path.splitext(image_file)[0] + ".txt")
                        dest_image = os.path.join(img_dir, image_file)
                        shutil.move(image_path, dest_image)
                        if os.path.exists(caption_file):
                            dest_caption = os.path.join(img_dir, os.path.splitext(image_file)[0] + ".txt")
                            shutil.move(caption_file, dest_caption)
            # Do NOT move or modify regularization images here; they are already copied by _process_regularization.
            # Only generate the Kohya config and leave regularization images as they are.
            self.message.emit("Kohya SS project structure created successfully")
        except Exception as e:
            self.message.emit(f"Error creating Kohya SS structure: {e}")
            self.logger.info("Error creating Kohya SS structure: %s", str(e))

    def _generate_kohya_config(self, training_folders):
        """Generate Kohya SS configuration files (JSON only, no TOML)."""
        try:
            config = {
                "general": {
                    "enable_bucket": True,
                    "caption_extension": ".txt",
                    "shuffle_caption": True,
                    "keep_tokens": 0,
                    "max_token_length": 225,
                    "bucket_reso_steps": 64,
                    "bucket_no_upscale": False
                },
                "datasets": []
            }
            # Add training dataset configuration
            if training_folders:
                for entry, folder_name in training_folders.items():
                    img_folder = os.path.join("img", folder_name)
                    dataset_config = {
                        "resolution": 512,
                        "batch_size": 1,
                        "enable_bucket": True,
                        "dreambooth": True if self.training_mode == "LoRA/DreamBooth" else False,
                        "image_dir": img_folder,
                        "caption_extension": ".txt",
                        "class_tokens": entry.class_prompt or self.class_prompt if self.training_mode == "LoRA/DreamBooth" else None,
                        "instance_tokens": entry.instance_prompt or self.instance_prompt if self.training_mode == "LoRA/DreamBooth" else None,
                        "num_repeats": entry.repeats or self.repeats
                    }
                    # Add regularization if it exists
                    if os.path.exists(os.path.join(self.dest, "regularization")):
                        class_prompt = entry.class_prompt or self.class_prompt
                        class_folder_name = f"{self.repeats}_{class_prompt}"
                        class_folder_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in class_folder_name)
                        class_folder_name = "_".join(class_folder_name.split())
                        reg_folder = os.path.join("regularization", class_folder_name)
                        if os.path.exists(os.path.join(self.dest, reg_folder)):
                            dataset_config["reg_img_dir"] = reg_folder
                    config["datasets"].append(dataset_config)
            # Save config file as JSON only
            config_path = os.path.join(self.dest, "dataset_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.message.emit(f"Configuration saved as {config_path}")
            self.logger.info("Kohya SS config generated: %s", config_path)
        except Exception as e:
            self.message.emit(f"Error generating config: {e}")
            self.logger.info("Error generating Kohya SS config: %s", str(e))

class FolderDropWidget(QtWidgets.QWidget):
    folderDropped = QtCore.pyqtSignal(str)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.folderDropped.emit(path)


class FolderWidget(QtWidgets.QWidget):
    def __init__(self, entry: FolderEntry, parent=None, remove_callback=None, get_total_images=None, get_total_megapixels=None, get_unlock_state=None):
        super().__init__(parent)
        self.entry = entry
        self.remove_callback = remove_callback
        self.get_total_images = get_total_images
        self.get_total_megapixels = get_total_megapixels
        self.get_unlock_state = get_unlock_state
        
        # Main layout with ultra-minimal margins
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # No margins at all
        main_layout.setSpacing(0)  # No spacing between frame elements
        
        # Precompute a maximally distinct pastel palette (24 colors, interleaved)
        def get_distinct_pastel_palette():
            # Interleave the color wheel for max contrast
            base = list(range(24))
            order = []
            step = 12
            used = set()
            for i in range(24):
                idx = (i * step) % 24
                while idx in used:
                    idx = (idx + 1) % 24
                order.append(idx)
                used.add(idx)
            palette = []
            for idx in order:
                hue = idx / 24.0
                saturation = 0.9  # Increased for more vividness
                lightness = 0.85
                r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
                r = int(r * 255)
                g = int(g * 255)
                b = int(b * 255)
                palette.append(f"rgba({r}, {g}, {b}, 0.25)")
            return palette
        PASTEL_PALETTE = get_distinct_pastel_palette()

        # Assign color based on folder path hash
        if entry.path:
            idx = abs(hash(entry.path)) % len(PASTEL_PALETTE)
            folder_color = PASTEL_PALETTE[idx]
        else:
            folder_color = "rgba(224, 224, 224, 0.25)"  # Light gray for no folder
        
        # Create a colored frame that contains everything
        self.folder_frame = QtWidgets.QFrame()
        
        # Set the frame with a very visible colored background and minimal border
        self.folder_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {folder_color};
                border: 1px solid {folder_color};  /* Minimal border */
                border-radius: 3px;  /* Smaller radius */
                margin: 1px;  /* Minimal margin */
                padding: 3px;  /* Minimal padding */
            }}
            QLabel {{
                background-color: rgba(255, 255, 255, 220);
                border: 1px solid #AAAAAA;
                border-radius: 3px;  /* Reduced from 4px */
                padding: 2px 4px;  /* Reduced from 3px 6px */
                color: #333333;
            }}
            QSpinBox, QLineEdit {{
                background-color: white;
                border: 1px solid #AAAAAA;  /* Reduced from 2px */
                border-radius: 3px;  /* Reduced from 4px */
                padding: 2px;  /* Reduced from 3px */
            }}
            QPushButton {{
                background-color: white;
                border: 1px solid #AAAAAA;  /* Reduced from 2px */
                border-radius: 4px;  /* Reduced from 6px */
                padding: 4px 8px;  /* Reduced from 6px 12px */
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #F0F0F0;
                border: 1px solid #888888;  /* Reduced from 2px */
            }}
            QGroupBox {{
                background-color: rgba(255, 255, 255, 180);
                border: 2px solid #777777;  /* Reduced from 3px */
                border-radius: 6px;  /* Reduced from 8px */
                margin-top: 8px;  /* Reduced from 15px */
                padding-top: 8px;  /* Reduced from 15px */
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;  /* Reduced from 15px */
                padding: 0 6px 0 6px;  /* Reduced from 0 8px 0 8px */
                background-color: white;
                border: 1px solid #777777;  /* Reduced from 2px */
                border-radius: 3px;  /* Reduced from 4px */
                color: #333333;
            }}
        """)
        
        # Layout inside the colored frame
        frame_layout = QtWidgets.QVBoxLayout(self.folder_frame)
        frame_layout.setContentsMargins(15, 15, 15, 15)
        frame_layout.setSpacing(8)

        # Top row: folder path, browse, remove
        top_row = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLabel(entry.path or "<No folder selected>")
        self.path_label.setMinimumWidth(200)
        self.path_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #222222;")
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.setToolTip("Select or change the folder for this section.")
        self.browse_btn.clicked.connect(self.browse_folder)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Remove this folder section.")
        self.remove_btn.clicked.connect(self.remove_self)
        top_row.addWidget(self.path_label)
        top_row.addWidget(self.browse_btn)
        top_row.addWidget(self.remove_btn)
        frame_layout.addLayout(top_row)

        # Megapixel indicator label (total available)
        self.megapixel_label = QtWidgets.QLabel()
        frame_layout.addWidget(self.megapixel_label)
        self.image_count = 0
        self.megapixels = 0
        self._mp_cache_valid = False
        self.update_megapixel_label()

        # MP Utilized label (color-coded, updated by set_megapixel_used_label)
        self.mp_utilized_label = QtWidgets.QLabel()
        frame_layout.addWidget(self.mp_utilized_label)
        
        # Images to be sampled indicator
        self.images_sampled_label = QtWidgets.QLabel()
        frame_layout.addWidget(self.images_sampled_label)

        # Megapixel percent used label (now shows MP used and percent of total, color-coded)
        self.megapixel_percent_label = QtWidgets.QLabel()
        # Do NOT add self.megapixel_percent_label to frame_layout (remove or comment out the line)
        # frame_layout.addWidget(self.megapixel_percent_label)

        # Percent parameters stacked vertically, ultra-compact layout
        percent_layout = QtWidgets.QVBoxLayout()
        percent_layout.setSpacing(0)  # No spacing between rows
        percent_layout.setContentsMargins(0, 0, 0, 0)  # No margins

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(2)  # Minimal spacing between label and spinbox
        row1.setContentsMargins(0, 0, 0, 0)  # No margins
        lbl1 = QtWidgets.QLabel("Folder %")
        self.percent_spin = QtWidgets.QSpinBox()
        self.percent_spin.setRange(0, 100)
        self.percent_spin.setValue(entry.percent)
        self.percent_spin.setToolTip("Percentage of total images to sample from this folder (used in Total Number mode).")
        self.percent_spin.setFixedWidth(55)  # Increased width for 3 digits
        self.percent_spin.setMaximumHeight(18)  # Smaller height
        row1.addWidget(lbl1)
        row1.addWidget(self.percent_spin)
        row1.addStretch(1)
        percent_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.blank_label = QtWidgets.QLabel("Blank")
        self.blank_spin = QtWidgets.QSpinBox()
        self.blank_spin.setRange(0, 100)
        self.blank_spin.setValue(entry.blank)
        self.blank_spin.setToolTip("% of images with blank captions.")
        self.blank_spin.setFixedWidth(50)
        row2.addWidget(self.blank_label)
        row2.addWidget(self.blank_spin)
        row2.addStretch(1)
        percent_layout.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        self.basic_label = QtWidgets.QLabel("Basic")
        self.basic_spin = QtWidgets.QSpinBox()
        self.basic_spin.setRange(0, 100)
        self.basic_spin.setValue(entry.basic)
        self.basic_spin.setToolTip("% of images with basic captions.")
        self.basic_spin.setFixedWidth(50)
        row3.addWidget(self.basic_label)
        row3.addWidget(self.basic_spin)
        row3.addStretch(1)
        percent_layout.addLayout(row3)

        row4 = QtWidgets.QHBoxLayout()
        self.detailed_label = QtWidgets.QLabel("Detailed")
        self.detailed_spin = QtWidgets.QSpinBox()
        self.detailed_spin.setRange(0, 100)
        self.detailed_spin.setValue(entry.detailed)
        self.detailed_spin.setToolTip("% of images with detailed captions.")
        self.detailed_spin.setFixedWidth(50)
        row4.addWidget(self.detailed_label)
        row4.addWidget(self.detailed_spin)
        row4.addStretch(1)
        percent_layout.addLayout(row4)

        row5 = QtWidgets.QHBoxLayout()
        self.structured_label = QtWidgets.QLabel("Structured")
        self.structured_spin = QtWidgets.QSpinBox()
        self.structured_spin.setRange(0, 100)
        self.structured_spin.setValue(entry.structured)
        self.structured_spin.setToolTip("% of images with structured captions.")
        self.structured_spin.setFixedWidth(50)
        row5.addWidget(self.structured_label)
        row5.addWidget(self.structured_spin)
        row5.addStretch(1)
        percent_layout.addLayout(row5)

        frame_layout.addLayout(percent_layout)

        # Caption percent total label
        self.caption_percent_label = QtWidgets.QLabel()
        frame_layout.addWidget(self.caption_percent_label)
        self.update_caption_percent_label()

        # Connect caption percent spinners to update label
        self.blank_spin.valueChanged.connect(self.update_caption_percent_label)
        self.basic_spin.valueChanged.connect(self.update_caption_percent_label)
        self.detailed_spin.valueChanged.connect(self.update_caption_percent_label)
        self.structured_spin.valueChanged.connect(self.update_caption_percent_label)

        # Connect percent spin to check max allowed
        self.percent_spin.valueChanged.connect(self.check_max_percent)

        self.check_max_percent()
        
        # Initial caption style color update
        self.update_caption_style_colors()
        
        # Initialize images sampled label
        self.set_images_sampled_label(0, self.image_count)

        # Training Settings section - readable but compact
        training_frame = QtWidgets.QGroupBox("Dataset Folder Naming")
        self.training_frame = training_frame  # Save reference for show/hide
        training_frame.setMaximumHeight(150)  # Increased from 120 to 150 to ensure Repeats field is visible
        training_layout = QtWidgets.QVBoxLayout(training_frame)
        training_layout.setSpacing(4)  # Increased from 0 for better readability
        training_layout.setContentsMargins(8, 8, 8, 8)  # Increased margins for readability
        
        # Instance prompt
        instance_layout = QtWidgets.QHBoxLayout()
        instance_layout.setSpacing(6)  # Increased spacing for readability
        instance_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        instance_layout.addWidget(QtWidgets.QLabel("Instance:"))
        self.instance_edit = QtWidgets.QLineEdit()
        self.instance_edit.setPlaceholderText("e.g., ohwx, sks (optional)")
        self.instance_edit.setText(entry.instance_prompt)
        self.instance_edit.setToolTip("Instance identifier for this folder's training images. Leave blank for no instance prompt.")
        self.instance_edit.textChanged.connect(self.update_entry)
        self.instance_edit.setMaximumHeight(24)  # Slightly larger input field
        instance_layout.addWidget(self.instance_edit)
        training_layout.addLayout(instance_layout)
        
        # Class prompt and repeats on same row
        class_repeats_layout = QtWidgets.QHBoxLayout()
        class_repeats_layout.setSpacing(6)  # Increased spacing
        
        # Class prompt
        class_repeats_layout.addWidget(QtWidgets.QLabel("Class:"))
        self.class_edit = QtWidgets.QLineEdit()
        self.class_edit.setPlaceholderText("e.g., person, woman, man")
        self.class_edit.setText(entry.class_prompt)
        self.class_edit.setToolTip("Class/category for regularization images.")
        self.class_edit.textChanged.connect(self.update_entry)
        self.class_edit.setMaximumHeight(24)  # Slightly larger input field
        class_repeats_layout.addWidget(self.class_edit)
        
        # Repeats
        class_repeats_layout.addWidget(QtWidgets.QLabel("Repeats:"))
        self.repeats_spin = QtWidgets.QSpinBox()
        self.repeats_spin.setRange(1, 100)
        self.repeats_spin.setValue(entry.repeats)
        self.repeats_spin.setToolTip("Number of training repeats for this folder.")
        self.repeats_spin.valueChanged.connect(self.update_entry)
        self.repeats_spin.setMaximumWidth(60)  # Fixed width for repeats
        self.repeats_spin.setMaximumHeight(24)  # Match other inputs
        class_repeats_layout.addWidget(self.repeats_spin)
        
        training_layout.addLayout(class_repeats_layout)

        frame_layout.addWidget(training_frame)

        # Add the colored frame to the main layout
        main_layout.addWidget(self.folder_frame)

        # Connect training settings signals
        self.instance_edit.textChanged.connect(self.update_entry)
        self.class_edit.textChanged.connect(self.update_entry)
        self.repeats_spin.valueChanged.connect(self.update_entry)

    def update_image_count_and_megapixels(self, force=False):
        if self._mp_cache_valid and not force:
            return
        self.image_count = 0
        self.megapixels = 0
        if self.entry.path and os.path.isdir(self.entry.path):
            for f in os.listdir(self.entry.path):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    self.image_count += 1
                    try:
                        img = Image.open(os.path.join(self.entry.path, f))
                        w, h = img.size
                        self.megapixels += (w * h) / 1_000_000
                    except Exception:
                        pass
        self._mp_cache_valid = True
        self.update_megapixel_label()

    def update_megapixel_label(self):
        self.megapixel_label.setText(f"Total: <b>{self.megapixels:.2f} MP</b>")

    def get_folder_megapixels(self):
        self.update_image_count_and_megapixels()
        return self.megapixels

    def check_max_percent(self):
        unlock_state = self.get_unlock_state() if self.get_unlock_state else False
        print(f"[DEBUG] check_max_percent for {self.entry.path}, unlock_state: {unlock_state}")
        
        # If percentages are unlocked, don't enforce limits
        if unlock_state:
            self.percent_spin.setStyleSheet('color: black;')
            self.percent_spin.setToolTip("Percentage of total images to sample from this folder (independent mode).")
            print(f"[DEBUG] Set to independent mode for {self.entry.path}")
            return
            
        # When locked, enforce limits based on available images
        total_images = self.get_total_images() if self.get_total_images else 1
        self.update_image_count_and_megapixels()
        if total_images == 0:
            max_percent = 0
        else:
            max_percent = int((self.image_count / total_images) * 100)
        
        current_percent = self.percent_spin.value()
        print(f"[DEBUG] Locked mode for {self.entry.path}: current={current_percent}, max={max_percent}, images={self.image_count}")
        
        # Reset style first to ensure clean state
        self.percent_spin.setStyleSheet('')
        
        if current_percent > max_percent:
            self.percent_spin.setStyleSheet('color: red;')
            self.percent_spin.setToolTip(f"This folder only has {self.image_count} images. Max allowed percent is {max_percent} for total {total_images} images.")
            # Only auto-adjust if we're switching from unlocked to locked mode
            if max_percent >= 0:
                self.percent_spin.setValue(max_percent)
            print(f"[DEBUG] Set to red (over limit) for {self.entry.path}")
        elif self.image_count == 0:
            self.percent_spin.setStyleSheet('color: red;')
            self.percent_spin.setToolTip("This folder has no images.")
            print(f"[DEBUG] Set to red (no images) for {self.entry.path}")
        else:
            self.percent_spin.setStyleSheet('color: black;')
            self.percent_spin.setToolTip(f"Percentage of total images to sample from this folder (max {max_percent} for {self.image_count} images).")
            print(f"[DEBUG] Set to black (valid) for {self.entry.path}")

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder and folder != self.entry.path:
            self.entry.path = folder
            self.path_label.setText(folder)
            
            # Update the colored background for the new folder
            import random
            
            def generate_random_color(seed_value, idx=0):
                # Quantize hue into 24 buckets (every 15 degrees)
                hue_bucket = abs(seed_value) % 24
                hue = hue_bucket / 24.0
                saturation = 0.75
                # Alternate lightness for even/odd idx
                lightness = 0.7 if idx % 2 == 0 else 0.8
                r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
                r = int(r * 255)
                g = int(g * 255)
                b = int(b * 255)
                return f"rgba({r}, {g}, {b}, 0.25)"
            
            folder_color = generate_random_color(hash(folder), hash(self))
            self._folder_color = folder_color
            
            # Update the frame background color with improved styling
            self.folder_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {folder_color};
                    border: 1px solid {folder_color};  /* Minimal border */
                    border-radius: 3px;  /* Smaller radius */
                    margin: 1px;  /* Minimal margin */
                    padding: 3px;  /* Minimal padding */
                }}
                QLabel {{
                    background-color: rgba(255, 255, 255, 220);
                    border: 1px solid #AAAAAA;
                    border-radius: 2px;  /* Smaller radius */
                    padding: 1px 2px;  /* Minimal padding */
                    color: #333333;
                    font-size: 10px;  /* Readable font size */
                }}
                QSpinBox, QLineEdit {{
                    background-color: white;
                    border: 2px solid #CCCCCC;
                    border-radius: 3px;
                    padding: 2px 4px;  /* Adequate padding */
                    font-size: 10px;  /* Readable font size */
                    min-height: 20px;  /* Minimum readable height */
                }}
                QGroupBox {{
                    font-weight: bold;
                    border: 2px solid #AAAAAA;
                    border-radius: 5px;
                    margin-top: 8px;
                    padding-top: 4px;
                    font-size: 10px;  /* Readable font size */
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px 0 4px;
                    font-size: 10px;  /* Readable font size */
                }}
                QPushButton {{
                    background-color: #F0F0F0;
                    border: 2px solid #CCCCCC;
                    border-radius: 4px;
                    padding: 3px 8px;  /* Adequate padding */
                    font-size: 10px;  /* Readable font size */
                    min-height: 18px;  /* Minimum button height */
                }}
                QPushButton:hover {{
                    background-color: #E0E0E0;
                    border-color: #AAAAAA;
                }}
            """)
            
            self.update_image_count_and_megapixels(force=True)
            self.check_max_percent()
            self.update_caption_style_colors()

    def set_blank_enabled(self, enabled: bool):
        self.blank_spin.setEnabled(enabled)
        if enabled:
            self.blank_label.setStyleSheet("color: green;")
        else:
            self.blank_label.setStyleSheet("color: gray;")
        self.update_caption_percent_label()

    def update_caption_percent_label(self):
        # Only include blank if enabled
        blank_val = self.blank_spin.value() if self.blank_spin.isEnabled() else 0
        total = blank_val + self.basic_spin.value() + self.detailed_spin.value() + self.structured_spin.value()
        color = 'green' if total == 100 else 'red'
        self.caption_percent_label.setText(f"Caption % Total: <b><span style='color:{color}'>{total}</span></b>")
        self.update_caption_style_colors()

    def update_caption_style_colors(self):
        if not self.entry.path or not os.path.isdir(self.entry.path):
            for label in [self.blank_label, self.basic_label, self.detailed_label, self.structured_label]:
                label.setStyleSheet("color: gray;")
            return
        image_names = set()
        for f in os.listdir(self.entry.path):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                name = os.path.splitext(f)[0]
                image_names.add(name)
        total_images = len(image_names)
        caption_styles = [
            ("blank", self.blank_label),
            ("basic", self.basic_label), 
            ("detailed", self.detailed_label),
            ("structured", self.structured_label)
        ]
        for style_name, label in caption_styles:
            if style_name == "blank":
                if not self.blank_spin.isEnabled():
                    label.setStyleSheet("color: gray;")
                    label.setToolTip("Blank captions are disabled in this mode.")
                    continue
                label.setStyleSheet("color: green;")
                label.setToolTip("Blank captions (always available)")
                continue
            subfolder_path = os.path.join(self.entry.path, style_name)
            caption_count = 0
            if os.path.isdir(subfolder_path):
                for name in image_names:
                    caption_file = os.path.join(subfolder_path, f"{name}.txt")
                    if os.path.exists(caption_file):
                        caption_count += 1
            else:
                traditional_caption_count = 0
                for name in image_names:
                    caption_file = os.path.join(self.entry.path, f"{name}.txt")
                    if os.path.exists(caption_file):
                        traditional_caption_count += 1
                if traditional_caption_count > 0:
                    caption_count = traditional_caption_count
            if caption_count == 0:
                color = "red"
                tooltip = f"No {style_name} captions found"
            elif caption_count == total_images:
                color = "green" 
                tooltip = f"All {total_images} {style_name} captions available"
            else:
                color = "orange"
                tooltip = f"Only {caption_count}/{total_images} {style_name} captions available"
            label.setStyleSheet(f"color: {color};")
            label.setToolTip(tooltip)

    def remove_self(self):
        print("[DEBUG] Remove button pressed for", self.entry.path)
        if self.remove_callback:
            self.remove_callback(self)

    def update_entry(self):
        self.entry.percent = self.percent_spin.value()
        self.entry.blank = self.blank_spin.value()
        self.entry.basic = self.basic_spin.value()
        self.entry.detailed = self.detailed_spin.value()
        self.entry.structured = self.structured_spin.value()
        self.entry.instance_prompt = self.instance_edit.text()
        self.entry.class_prompt = self.class_edit.text()
        self.entry.repeats = self.repeats_spin.value()

    def set_megapixel_used_label(self, est_mp_used, color):
        self.mp_utilized_label.setText(f"MP Utilized: <b><span style='color:{color}'>{est_mp_used:.2f} MP</span></b>")
    
    def set_images_sampled_label(self, images_to_sample, total_images):
        """Update the label showing how many images will be sampled from this folder"""
        # Get unlock state from parent/main window if possible
        unlock_state = False
        if hasattr(self, 'get_unlock_state') and self.get_unlock_state:
            unlock_state = self.get_unlock_state()
        if total_images == 0:
            self.images_sampled_label.setText("Images to Sample: <b><span style='color:gray'>0 / 0</span></b>")
        else:
            percentage = (images_to_sample / total_images) * 100 if total_images > 0 else 0
            # Color coding based on sampling percentage
            if percentage == 0:
                color = 'gray'
            elif unlock_state:
                # In unlock mode, use green for any nonzero percentage
                color = 'green'
            elif percentage <= 50:
                color = 'green'
            elif percentage <= 80:
                color = 'orange'
            else:
                color = 'red'
            self.images_sampled_label.setText(f"Images to Sample: <b><span style='color:{color}'>{images_to_sample} / {total_images} ({percentage:.1f}%)</span></b>")

    def set_training_fields_visible(self, visible: bool):
        """Show/hide and enable/disable instance, class, and repeats fields for training mode switching."""
        self.training_frame.setVisible(visible)
        self.instance_edit.setVisible(visible)
        self.class_edit.setVisible(visible)
        self.repeats_spin.setVisible(visible)
        self.instance_edit.setEnabled(visible)
        self.class_edit.setEnabled(visible)
        self.repeats_spin.setEnabled(visible)

    def get_effective_caption_total(self):
        """Return the sum of only enabled caption types (exclude blank if disabled)"""
        if self.blank_spin.isEnabled():
            return self.blank_spin.value() + self.basic_spin.value() + self.detailed_spin.value() + self.structured_spin.value()
        else:
            return self.basic_spin.value() + self.detailed_spin.value() + self.structured_spin.value()

# --- Custom dialog for multi-folder selection ---
class MultiFolderSelectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Select Multiple Folders")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)

        # Address bar
        self.address_bar = QtWidgets.QLineEdit()
        self.address_bar.setPlaceholderText("Enter folder path and press Enter...")
        layout.addWidget(self.address_bar)
        self.address_bar.installEventFilter(self)

        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath("")
        self.model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Drives)
        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.model)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        
        # Set initial directory if provided
        if initial_dir and os.path.isdir(initial_dir):
            initial_index = self.model.index(initial_dir)
            if initial_index.isValid():
                self.tree.setRootIndex(initial_index)
                self.address_bar.setText(initial_dir)
            else:
                self.tree.setRootIndex(self.model.index(self.model.rootPath()))
        else:
            self.tree.setRootIndex(self.model.index(self.model.rootPath()))
        
        # Hide columns except for folder name
        for i in range(1, self.model.columnCount()):
            self.tree.hideColumn(i)
        layout.addWidget(self.tree)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Address bar navigation
        self.address_bar.returnPressed.connect(self.navigate_to_path)
        self.tree.selectionModel().currentChanged.connect(self.update_address_bar)
        self.tree.expanded.connect(self.update_address_bar)
        self.tree.clicked.connect(self.update_address_bar)

    def eventFilter(self, obj, event):
        if obj is self.address_bar and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.navigate_to_path()
                return True  # Block further processing (prevents dialog accept)
        return super().eventFilter(obj, event)

    def navigate_to_path(self):
        path = self.address_bar.text().strip()
        if os.path.isdir(path):
            idx = self.model.index(path)
            if idx.isValid():
                self.tree.setRootIndex(idx)
                self.tree.selectionModel().clearSelection()
                self.tree.scrollTo(idx)
                self.tree.setCurrentIndex(idx)
        else:
            QtWidgets.QMessageBox.warning(self, "Invalid Path", f"'{path}' is not a valid folder.")

    def update_address_bar(self, *args):
        idx = self.tree.currentIndex()
        if idx.isValid():
            self.address_bar.setText(self.model.filePath(idx))

    def selected_folders(self):
        indexes = self.tree.selectionModel().selectedRows(0)
        # Only return directories
        return [self.model.filePath(idx) for idx in indexes if self.model.isDir(idx)]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # (Removed creation of model, log, and config folders)
        super().__init__()
        
        # Set window title
        self.setWindowTitle("Kohya SS Training Dataset Composer")
        
        # Set compact window size and center it (responsive to screen size)
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.35)  # 35% of screen width (was 17.5%)
        height = int(screen.height() * 0.85)  # 85% of screen height
        self.resize(width, height)
        self.move(
            screen.left() + (screen.width() - width) // 2,
            screen.top() + (screen.height() - height) // 2
        )
        self.entries: List[FolderEntry] = []
        self.folder_widgets = []
        self.settings = QtCore.QSettings("sampler", "dataset")
        self.last_folder_path = ""  # Remember last folder location for Add Multiple dialog
        self.convert_to_png = True  # default value
        self.lora_regularization_enabled = None  # Track LoRA/DreamBooth regularization state

        # Add menu bar with Edit > Refresh Directories
        menubar = self.menuBar()
        edit_menu = menubar.addMenu("Edit")
        refresh_action = QtWidgets.QAction("Refresh Directories", self)
        refresh_action.setShortcut("Ctrl+R")
        refresh_action.triggered.connect(self.refresh_directories)
        edit_menu.addAction(refresh_action)

        # --- Refactor main layout to three columns ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_hbox = QtWidgets.QHBoxLayout(central)
        left_vbox = QtWidgets.QVBoxLayout()
        middle_vbox = QtWidgets.QVBoxLayout()
        right_vbox = QtWidgets.QVBoxLayout()

        # --- Create all regularization widgets before any right column layout code ---
        self.reg_folder_edit = QtWidgets.QLineEdit()
        self.reg_folder_edit.setPlaceholderText("Select regularization folder...")
        self.regularization_checkbox = QtWidgets.QCheckBox("Enable Regularization")
        self.regularization_checkbox.setToolTip("Add regularization images to improve training stability")
        self.reg_percent_spin = QtWidgets.QSpinBox()
        self.reg_percent_spin.setRange(0, 10000)
        self.reg_percent_spin.setValue(0)
        self.reg_percent_spin.setSuffix("%")
        self.reg_percent_spin.setToolTip("Percentage of regularization images to use (0-10000%). You can set values above 100% if desired.")
        self.reg_folders_widget = QtWidgets.QWidget()
        self.reg_folders_layout = QtWidgets.QVBoxLayout(self.reg_folders_widget)
        self.reg_folders_layout.setContentsMargins(0, 0, 0, 0)
        self.reg_folders_layout.setSpacing(2)
        self.reg_scroll = QtWidgets.QScrollArea()
        self.reg_scroll.setWidgetResizable(True)
        self.reg_scroll.setWidget(self.reg_folders_widget)
        self.reg_scroll.setMaximumHeight(16777215)
        self.reg_scroll.setMinimumHeight(0)
        self.reg_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.reg_select_all_btn = QtWidgets.QPushButton("Select All")
        self.reg_select_none_btn = QtWidgets.QPushButton("Select None")
        self.reg_summary_label = QtWidgets.QLabel("")
        # Now proceed with right_vbox layout as before, referencing these widgets

        # Step 1: Kohya SS Project Setup
        step1_frame = QtWidgets.QGroupBox("Step 1: Kohya SS Project Setup")
        step1_layout = QtWidgets.QVBoxLayout(step1_frame)
        
        # Training mode selection
        training_mode_layout = QtWidgets.QHBoxLayout()
        training_mode_layout.addWidget(QtWidgets.QLabel("Training Mode:"))
        self.training_mode_combo = QtWidgets.QComboBox()
        self.training_mode_combo.addItems(["LoRA/DreamBooth", "Fine-tuning"])
        self.training_mode_combo.setToolTip("LoRA/DreamBooth: Character/subject training. Fine-tuning: Style/concept training.")
        training_mode_layout.addWidget(self.training_mode_combo)
        training_mode_layout.addStretch(1)
        step1_layout.addLayout(training_mode_layout)
        
        left_vbox.addWidget(step1_frame)
        
        # Connect training mode to update UI visibility
        self.training_mode_combo.currentTextChanged.connect(self.update_regularization_for_training_mode)

        # Step 2: Folder Setup
        step2_frame = QtWidgets.QGroupBox("Step 2: Folder Setup")
        step2_layout = QtWidgets.QVBoxLayout(step2_frame)
        
        # Destination folder
        dest_layout = QtWidgets.QHBoxLayout()
        dest_layout.addWidget(QtWidgets.QLabel("Destination:"))
        self.dest_edit = QtWidgets.QLineEdit()
        self.dest_btn = QtWidgets.QPushButton("Browse")
        self.dest_btn.clicked.connect(self.browse_dest)
        dest_layout.addWidget(self.dest_edit)
        dest_layout.addWidget(self.dest_btn)
        step2_layout.addLayout(dest_layout)
        # (Remove all regularization-related controls from here)
        left_vbox.addWidget(step2_frame)

        # Step 3: Training Data Setup
        step3_frame = QtWidgets.QGroupBox("Step 3: Training Data Setup")
        step3_layout = QtWidgets.QVBoxLayout(step3_frame)
        
        # Total images and increase controls
        total_layout = QtWidgets.QHBoxLayout()
        total_layout.addWidget(QtWidgets.QLabel("Total Images:"))
        self.total_spin = QtWidgets.QSpinBox()
        self.total_spin.setMaximum(1_000_000)
        self.total_spin.setValue(1000)
        self.total_spin.setFixedWidth(100)
        total_layout.addWidget(self.total_spin)
        # Add total sampled images indicator
        self.sampled_images_label = QtWidgets.QLabel()
        total_layout.addWidget(self.sampled_images_label)
        self.available_images_label = QtWidgets.QLabel()
        total_layout.addWidget(self.available_images_label)
        total_layout.addStretch(1)
        step3_layout.addLayout(total_layout)
        # Update sampled images indicator whenever relevant
        self.total_spin.valueChanged.connect(self.update_sampled_images_label)
        # Will also connect percent_spin changes in add_folder and _load_settings

        # --- Resize to N Pixels Checkbox and Spinbox ---
        resize_layout = QtWidgets.QHBoxLayout()
        self.resize_checkbox = QtWidgets.QCheckBox("Resize to")
        self.resize_checkbox.setChecked(True)
        self.resize_spin = QtWidgets.QSpinBox()
        self.resize_spin.setRange(64, 8192)
        self.resize_spin.setValue(1024)
        self.resize_spin.setFixedWidth(70)
        resize_layout.addWidget(self.resize_checkbox)
        resize_layout.addWidget(self.resize_spin)
        resize_layout.addWidget(QtWidgets.QLabel("pixels (shortest side)"))
        resize_layout.addStretch(1)
        step3_layout.addLayout(resize_layout)

        # PNG conversion checkbox
        self.png_checkbox = QtWidgets.QCheckBox("Convert non-PNG images to PNG on save")
        self.png_checkbox.setChecked(True)
        self.png_checkbox.setToolTip("If checked, all non-PNG images will be saved as PNG in the output folder.")
        step3_layout.addWidget(self.png_checkbox)
        
        # Copy Training Set / Copy Regularization Set checkboxes
        copy_layout = QtWidgets.QHBoxLayout()
        self.copy_training_checkbox = QtWidgets.QCheckBox("Copy Training Set")
        self.copy_training_checkbox.setChecked(True)
        self.copy_training_checkbox.setToolTip("Copy training images to Kohya SS project structure")
        copy_layout.addWidget(self.copy_training_checkbox)
        
        self.copy_regularization_checkbox = QtWidgets.QCheckBox("Copy Regularization Set")
        self.copy_regularization_checkbox.setChecked(True)
        self.copy_regularization_checkbox.setToolTip("Copy regularization images to Kohya SS project structure")
        copy_layout.addWidget(self.copy_regularization_checkbox)
        copy_layout.addStretch(1)
        step3_layout.addLayout(copy_layout)
        
        # Training Folder Seed Controls
        seed_frame = QtWidgets.QGroupBox("Training Folder Seed Controls")
        seed_layout = QtWidgets.QVBoxLayout(seed_frame)
        
        # Training and Regularization seed controls in a grid for perfect alignment
        seed_grid = QtWidgets.QGridLayout()
        seed_grid.addWidget(QtWidgets.QLabel("Image Sample Seed:"), 0, 0)
        self.training_seed_combo = QtWidgets.QComboBox()
        self.training_seed_combo.addItems(["Random", "Static"])
        self.training_seed_combo.setToolTip("Random: Generate new seed each time. Static: Use fixed seed value.")
        seed_grid.addWidget(self.training_seed_combo, 0, 1)
        self.training_seed_spin = QtWidgets.QSpinBox()
        self.training_seed_spin.setRange(0, 2147483647)
        self.training_seed_spin.setValue(random.randint(0, 999999))
        self.training_seed_spin.setToolTip("Seed value for random selection of training images")
        seed_grid.addWidget(self.training_seed_spin, 0, 2)
        seed_grid.addWidget(QtWidgets.QLabel("Regularization Sample Seed:"), 1, 0)
        self.reg_seed_combo = QtWidgets.QComboBox()
        self.reg_seed_combo.addItems(["Random", "Static"])
        self.reg_seed_combo.setToolTip("Random: Generate new seed each time. Static: Use fixed seed value.")
        seed_grid.addWidget(self.reg_seed_combo, 1, 1)
        self.reg_seed_spin = QtWidgets.QSpinBox()
        self.reg_seed_spin.setRange(0, 2147483647)
        self.reg_seed_spin.setValue(random.randint(0, 999999))
        self.reg_seed_spin.setToolTip("Seed value for random selection of regularization images")
        seed_grid.addWidget(self.reg_seed_spin, 1, 2)
        seed_layout.addLayout(seed_grid)
        
        step3_layout.addWidget(seed_frame)
        
        # Unlock folder percentages checkbox
        self.unlock_percentages_checkbox = QtWidgets.QCheckBox("Unlock folder percentages (allow independent percentages)")
        self.unlock_percentages_checkbox.setToolTip("When checked, folder percentages don't need to sum to 100%")
        step3_layout.addWidget(self.unlock_percentages_checkbox)
        
        left_vbox.addWidget(step3_frame)

        # Step 4: Source Folders (middle column)
        step4_frame = QtWidgets.QGroupBox("Step 4: Source Folders")
        step4_layout = QtWidgets.QVBoxLayout(step4_frame)
        
        # Summary labels (restored)
        self.total_percent_label = QtWidgets.QLabel()
        step4_layout.addWidget(self.total_percent_label)
        self.avg_mp_label = QtWidgets.QLabel()
        step4_layout.addWidget(self.avg_mp_label)

        # Add increase/decrease percent buttons above the source folders scroll area
        incdec_layout = QtWidgets.QHBoxLayout()
        incdec_layout.addStretch(1)
        self.increase_all_btn = QtWidgets.QPushButton("Increase All %")
        self.decrease_all_btn = QtWidgets.QPushButton("Decrease All %")
        self.increase_all_btn.clicked.connect(lambda: self.adjust_all_folder_percents(1))
        self.decrease_all_btn.clicked.connect(lambda: self.adjust_all_folder_percents(-1))
        incdec_layout.addWidget(self.increase_all_btn)
        incdec_layout.addWidget(self.decrease_all_btn)
        incdec_layout.addStretch(1)
        step4_layout.addLayout(incdec_layout)

        # Scroll area for folder widgets
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(240)
        self.list_widget = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.list_widget)
        self.scroll.setWidget(self.list_widget)
        step4_layout.addWidget(self.scroll)

        # Add folder buttons (only Add Folder and Add Multiple)
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add Folder")
        self.add_multiple_btn = QtWidgets.QPushButton("Add Multiple")
        from PyQt5.QtWidgets import QSizePolicy
        self.add_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.add_multiple_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn_layout.addWidget(self.add_btn, 1)
        btn_layout.addWidget(self.add_multiple_btn, 1)
        btn_layout.addStretch(0)
        step4_layout.addLayout(btn_layout)
        
        middle_vbox.addWidget(step4_frame)
        middle_vbox.setStretchFactor(step4_frame, 2)

        # --- Regularization Controls (right column, above folders list) ---
        # Regularization Folder label and Browse button on the same row
        reg_folder_row = QtWidgets.QHBoxLayout()
        reg_folder_label = QtWidgets.QLabel("Regularization Folder:")
        reg_folder_row.addWidget(reg_folder_label)
        reg_folder_browse_btn = QtWidgets.QPushButton("Browse")
        reg_folder_browse_btn.clicked.connect(self.browse_regularization_folder)
        reg_folder_row.addWidget(reg_folder_browse_btn)
        reg_folder_row.addStretch(1)
        right_vbox.addLayout(reg_folder_row)
        # Path field on its own row, stretching to the right
        from PyQt5.QtWidgets import QSizePolicy
        self.reg_folder_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_vbox.addWidget(self.reg_folder_edit)
        self.reg_folder_edit.textChanged.connect(self.update_regularization_info)

        # Enable Regularization checkbox
        reg_enable_layout = QtWidgets.QHBoxLayout()
        self.regularization_checkbox = QtWidgets.QCheckBox("Enable Regularization")
        self.regularization_checkbox.setToolTip("Add regularization images to improve training stability")
        reg_enable_layout.addWidget(self.regularization_checkbox)
        right_vbox.addLayout(reg_enable_layout)
        self.regularization_checkbox.stateChanged.connect(self.toggle_regularization_controls)
        self.regularization_checkbox.stateChanged.connect(self.on_regularization_checkbox_changed)

        # Regularization Percentage spinbox
        reg_percent_layout = QtWidgets.QHBoxLayout()
        reg_percent_label = QtWidgets.QLabel("Regularization Percentage:")
        reg_percent_layout.addWidget(reg_percent_label)
        self.reg_percent_spin = QtWidgets.QSpinBox()
        self.reg_percent_spin.setRange(0, 10000)
        self.reg_percent_spin.setValue(0)
        self.reg_percent_spin.setSuffix("%")
        self.reg_percent_spin.setToolTip("Percentage of regularization images to use (0-10000%). You can set values above 100% if desired.")
        reg_percent_layout.addWidget(self.reg_percent_spin)
        reg_percent_layout.addStretch(1)
        right_vbox.addLayout(reg_percent_layout)
        self.reg_percent_spin.valueChanged.connect(self.update_regularization_info)

        # Regularization Folders scroll area (with checkboxes)
        right_vbox.addWidget(self.reg_scroll, stretch=1)
        # Summary label and select buttons pinned to bottom
        right_vbox.addWidget(self.reg_summary_label)
        # Center the Select All / Select None buttons in the regularization section
        reg_btns_layout = QtWidgets.QHBoxLayout()
        reg_btns_layout.addStretch(1)
        self.reg_select_all_btn = QtWidgets.QPushButton("Select All")
        self.reg_select_none_btn = QtWidgets.QPushButton("Select None")
        self.reg_select_all_btn.clicked.connect(self.select_all_reg_folders)
        self.reg_select_none_btn.clicked.connect(self.select_none_reg_folders)
        reg_btns_layout.addWidget(self.reg_select_all_btn)
        reg_btns_layout.addWidget(self.reg_select_none_btn)
        reg_btns_layout.addStretch(1)
        right_vbox.addLayout(reg_btns_layout)
        right_vbox.addStretch(1)

        # Add left, middle, and right columns to main layout
        left_container = QtWidgets.QWidget()
        left_container.setMaximumWidth(460)
        left_container.setMinimumWidth(220)
        left_container.setLayout(left_vbox)
        main_hbox.addWidget(left_container)
        main_hbox.addLayout(middle_vbox, 4)
        main_hbox.addLayout(right_vbox, 2)

        # Unify font size in the regularization section (right column)
        regularization_style = """
            QLineEdit, QSpinBox, QComboBox, QLabel, QPushButton, QCheckBox {
                font-size: 11px;
            }
        """
        for widget in [self.reg_folder_edit, self.regularization_checkbox, self.reg_percent_spin, self.reg_scroll, self.reg_select_all_btn, self.reg_select_none_btn, self.reg_summary_label]:
            widget.setStyleSheet(regularization_style)

        # Step 5: Execute (Start button, log, progress)
        step5_frame = QtWidgets.QGroupBox("Step 5: Execute")
        step5_layout = QtWidgets.QVBoxLayout(step5_frame)
        
        # Remove or reduce stretch before log window
        # step5_layout.addStretch(1)  # Remove this line so log window can expand
        
        # Log box (expanding vertically)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Remove setFixedHeight

        # Progress bar below log window, matching width
        self.progress = QtWidgets.QProgressBar()
        self.progress.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Container for log and progress bar
        log_progress_container = QtWidgets.QWidget()
        log_progress_layout = QtWidgets.QVBoxLayout(log_progress_container)
        log_progress_layout.setContentsMargins(0, 0, 0, 0)
        log_progress_layout.setSpacing(2)
        log_progress_layout.addWidget(self.log_box)
        log_progress_layout.addWidget(self.progress)
        step5_layout.addWidget(log_progress_container, stretch=1)
        
        # Start button at the bottom
        self.start_btn = QtWidgets.QPushButton("Start")
        step5_layout.addWidget(self.start_btn)

        left_vbox.addWidget(step5_frame)
        left_vbox.setStretchFactor(step5_frame, 0)

        # Connect signals
        self.add_btn.clicked.connect(self.add_folder)
        self.add_multiple_btn.clicked.connect(self.add_multiple_folders)
        self.start_btn.clicked.connect(self.start_sampling)
        self.total_spin.valueChanged.connect(self.on_total_spin_changed)
        self.unlock_percentages_checkbox.stateChanged.connect(self.update_all_folder_max_percents)
        self.reg_percent_spin.valueChanged.connect(self.update_regularization_info)
        self.total_spin.valueChanged.connect(self.update_regularization_info)
        self.reg_percent_spin.valueChanged.connect(self.update_reg_summary)
        self.total_spin.valueChanged.connect(self.update_reg_summary)
        
        # Initialize
        self.reg_folder_checkboxes = {}  # Dictionary to store regularization folder checkbox widgets
        self.update_total_percent_label()
        self.update_avg_mp_label()

        # Restore settings after UI is built
        self._load_settings()

    def add_folder(self):
        # Prevent duplicate folder paths
        folder_paths = [e.path for e in self.entries]
        
        # Start from last used folder if available
        start_dir = self.last_folder_path if self.last_folder_path and os.path.isdir(self.last_folder_path) else ""
        
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if not folder or folder in folder_paths:
            if folder and folder in folder_paths:
                QtWidgets.QMessageBox.warning(self, "Duplicate Folder", "This folder has already been added.")
            return
        
        # Remember this folder's parent directory for next time
        self.last_folder_path = os.path.dirname(folder)
        
        entry = FolderEntry(path=folder)
        widget = FolderWidget(entry, remove_callback=self.remove_folder_widget, get_total_images=self.get_total_images, get_total_megapixels=self.get_total_megapixels, get_unlock_state=self.get_unlock_state)
        self.entries.append(entry)
        self.folder_widgets.append(widget)
        self.list_layout.addWidget(widget)
        widget.percent_spin.valueChanged.connect(self.update_total_percent_label)
        widget.percent_spin.valueChanged.connect(self.update_avg_mp_label)
        widget.percent_spin.valueChanged.connect(self.update_all_folder_mp_indicators)
        widget.percent_spin.valueChanged.connect(self.update_sampled_images_label)
        self.update_total_percent_label()
        self.update_all_folder_max_percents()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        self.update_available_images_label()

    def remove_folder_widget(self, widget):
        print("[DEBUG] remove_folder_widget called for", widget.entry.path)
        idx = self.folder_widgets.index(widget)
        self.folder_widgets.pop(idx)
        self.entries.pop(idx)
        for i in reversed(range(self.list_layout.count())):
            item = self.list_layout.itemAt(i)
            w = item.widget()
            self.list_layout.takeAt(i)
            if w is widget:
                w.setParent(None)
                w.deleteLater()
            elif w is not None:
                w.setParent(None)
        for w in self.folder_widgets:
            self.list_layout.addWidget(w)
        self.list_layout.update()
        self.list_layout.invalidate()
        self.list_widget.adjustSize()
        self.scroll.viewport().update()
        print("[DEBUG] List rebuilt and layout updated")
        self.update_total_percent_label()
        self.update_available_images_label()

    def update_total_percent_label(self):
        # Calculate the total number of images to be sampled from all folders
        sampled_total = 0
        for w in self.folder_widgets:
            w.update_image_count_and_megapixels()
            count = int(w.image_count * w.percent_spin.value() / 100)
            sampled_total += count
        target = self.total_spin.value()
        percent = int((sampled_total / target) * 100) if target > 0 else 0
        # If percentages are unlocked, don't enforce 100% rule
        if self.unlock_percentages_checkbox.isChecked():
            color = 'blue'  # Different color to indicate independent mode
            self.total_percent_label.setText(f"Total Number of Images % (Independent): <b><span style='color:{color}'>{percent}</span></b>")
        else:
            # Normal locked mode - enforce 100% rule
            if percent == 100:
                color = 'green'
            else:
                color = 'red'
            self.total_percent_label.setText(f"Total Number of Images %: <b><span style='color:{color}'>{percent}</span></b>")

    def browse_dest(self):
        start_dir = self.last_folder_path if self.last_folder_path and os.path.isdir(self.last_folder_path) else ""
        dest = QtWidgets.QFileDialog.getExistingDirectory(self, "Destination Folder", start_dir)
        if dest:
            self.dest_edit.setText(dest)
            self.last_folder_path = dest
            self.settings.setValue("last_folder_path", dest)

    def browse_regularization_folder(self):
        start_dir = self.last_folder_path if self.last_folder_path and os.path.isdir(self.last_folder_path) else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Regularization Folder", start_dir)
        if folder:
            self.reg_folder_edit.setText(folder)
            self.last_folder_path = folder
            self.settings.setValue("last_folder_path", folder)
            self.update_regularization_info()
            self.update_regularization_folders_list()

    def toggle_regularization_controls(self):
        """Enable/disable regularization controls based on checkbox state"""
        enabled = self.regularization_checkbox.isChecked()

        # Only enable/disable widgets that exist in the new UI
        self.reg_folders_widget.setEnabled(enabled)
        self.reg_scroll.setEnabled(enabled)
        self.reg_percent_spin.setEnabled(enabled)
        self.reg_summary_label.setEnabled(enabled)
        self.reg_select_all_btn.setEnabled(enabled)
        self.reg_select_none_btn.setEnabled(enabled)

        if enabled:
            self.update_regularization_info()
            self.update_regularization_folders_list()
        # Do NOT clear the list when disabling; just leave the checkboxes as they are.

    def update_regularization_info(self):
        """Update the regularization info display"""
        if not self.regularization_checkbox.isChecked():
            return
            
        reg_folder = self.reg_folder_edit.text()
        if not reg_folder or not os.path.isdir(reg_folder):
            return
        
        # Count images in regularization folder (including subfolders)
        total_reg_images = 0
        for root, dirs, files in os.walk(reg_folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    total_reg_images += 1
        
        # Calculate how many will be used
        main_dataset_size = self.total_spin.value()
        reg_percentage = self.reg_percent_spin.value()
        reg_images_to_use = int(main_dataset_size * reg_percentage / 100)
        
        # Update the regularization folder edit tooltip with this info
        if total_reg_images == 0:
            self.reg_folder_edit.setToolTip("No images found in folder")
        elif reg_images_to_use > total_reg_images:
            self.reg_folder_edit.setToolTip(f"Warning: Need {reg_images_to_use} images but only {total_reg_images} available")
        else:
            self.reg_folder_edit.setToolTip(f"Will use {reg_images_to_use} of {total_reg_images} available images")
        
        # Update regularization folders list
        self.update_regularization_folders_list()

    def update_regularization_for_training_mode(self):
        mode = self.training_mode_combo.currentText()
        enable_blank = (mode == "Fine-tuning")
        for w in self.folder_widgets:
            w.set_blank_enabled(enable_blank)
        if mode == "Fine-tuning":
            # In Fine-tuning mode, disable and gray out regularization controls
            self.regularization_checkbox.setStyleSheet('color: gray;')
            self.toggle_regularization_controls()
            for w in self.folder_widgets:
                w.set_training_fields_visible(False)
            self.copy_regularization_checkbox.setEnabled(False)
            self.copy_regularization_checkbox.setStyleSheet('color: gray;')
            self.reg_folders_widget.setEnabled(False)
            self.reg_scroll.setEnabled(False)
            self.reg_folders_widget.setStyleSheet('color: gray;')
            self.reg_scroll.setStyleSheet('color: gray;')
        else:
            # In LoRA/DreamBooth mode, enable regularization controls
            self.regularization_checkbox.setStyleSheet('')
            self.toggle_regularization_controls()
            for w in self.folder_widgets:
                w.set_training_fields_visible(True)
            self.copy_regularization_checkbox.setEnabled(True)
            self.copy_regularization_checkbox.setStyleSheet('')
            self.reg_folders_widget.setEnabled(True)
            self.reg_scroll.setEnabled(True)
            self.reg_folders_widget.setStyleSheet('')
            self.reg_scroll.setStyleSheet('')

    def on_regularization_checkbox_changed(self, state):
        # Only update LoRA state if in LoRA/DreamBooth mode
        if self.training_mode_combo.currentText() == "LoRA/DreamBooth":
            self.lora_regularization_enabled = bool(state)

    def start_sampling(self):
        for i in range(self.list_layout.count()):
            widget = self.list_layout.itemAt(i).widget()
            widget.update_entry()
        if self.entries:
            for i in range(self.list_layout.count()):
                self.list_layout.itemAt(i).widget().path_label.setStyleSheet("")
        # Only check percentage sum if unlock checkbox is NOT checked
        if not self.unlock_percentages_checkbox.isChecked():
            # Calculate the total number of images to be sampled from all folders
            sampled_total = 0
            for w in self.folder_widgets:
                w.update_image_count_and_megapixels()
                count = int(w.image_count * w.percent_spin.value() / 100)
                sampled_total += count
            target = self.total_spin.value()
            percent = int((sampled_total / target) * 100) if target > 0 else 0
            if percent != 100:
                QtWidgets.QMessageBox.warning(self, "Error", "Total Folder % must be 100 (based on the number of images to be sampled from all folders relative to the total number of images)")
                return
        for w in self.folder_widgets:
            if w.get_effective_caption_total() != 100:
                QtWidgets.QMessageBox.warning(self, "Error", f"Caption percentages for {w.entry.path} must sum to 100")
                return
        dest = self.dest_edit.text()
        if not dest:
            QtWidgets.QMessageBox.warning(self, "Error", "Destination folder required")
            return
        # 1. Ensure model, log, and config folders exist (create if missing, leave alone if present)
        for folder in ["model", "log", "config"]:
            os.makedirs(os.path.join(dest, folder), exist_ok=True)
        # 2. Check if img and/or regularization folders exist and are non-empty
        img_folder = os.path.join(dest, "img")
        reg_folder = os.path.join(dest, "regularization")
        img_exists = os.path.isdir(img_folder) and any(os.scandir(img_folder))
        reg_exists = os.path.isdir(reg_folder) and any(os.scandir(reg_folder))
        if img_exists or reg_exists:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("Destination Contains Training/Reg. Data")
            msg = []
            if img_exists:
                msg.append("'img' folder (training images)")
            if reg_exists:
                msg.append("'regularization' folder")
            msg_box.setText(f"The destination contains: {', '.join(msg)} with existing content.")
            msg_box.setInformativeText("What would you like to do?")
            concat_btn = msg_box.addButton("Add New Files (Keep Existing)", QtWidgets.QMessageBox.ActionRole)
            delete_btn = msg_box.addButton("Clean Training/Reg. Folders", QtWidgets.QMessageBox.DestructiveRole)
            cancel_btn = msg_box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            msg_box.setDefaultButton(concat_btn)
            msg_box.exec_()
            clicked_button = msg_box.clickedButton()
            if clicked_button == cancel_btn:
                return  # User cancelled
            elif clicked_button == delete_btn:
                deleted_items = 0
                if img_exists:
                    for item in os.listdir(img_folder):
                        item_path = os.path.join(img_folder, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                                deleted_items += 1
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                                deleted_items += 1
                        except Exception as e:
                            self.log_box.append(f"Failed to delete {item_path}: {e}")
                if reg_exists:
                    try:
                        shutil.rmtree(reg_folder)
                        deleted_items += 1
                    except Exception as e:
                        self.log_box.append(f"Failed to delete {reg_folder}: {e}")
                self.log_box.append(f"Cleaned {deleted_items} item(s) from 'img' and/or 'regularization' folders as selected. No other files or folders were deleted.")
        # Now create model, log, and config folders inside the destination
        for folder in ["model", "log", "config"]:
            os.makedirs(os.path.join(dest, folder), exist_ok=True)
        
        total = self.total_spin.value()
        self.convert_to_png = self.png_checkbox.isChecked()
        
        # Handle training seed
        training_seed = None
        if self.training_seed_combo.currentText() == "Random":
            training_seed = random.randint(0, 2147483647)
            self.training_seed_spin.setValue(training_seed)  # Update UI to show generated seed
            self.log_box.append(f"Generated training seed: {training_seed}")
        else:  # Static mode
            training_seed = self.training_seed_spin.value()
            self.log_box.append(f"Using training seed: {training_seed}")
        
        # Get regularization settings
        reg_folder = None
        reg_percent = 0
        reg_seed = None
        selected_reg_folders = None
        training_mode = self.training_mode_combo.currentText()
        if self.regularization_checkbox.isChecked() and training_mode != "Fine-tuning":
            reg_folder = self.reg_folder_edit.text()
            reg_percent = self.reg_percent_spin.value()
            if not reg_folder or not os.path.isdir(reg_folder):
                QtWidgets.QMessageBox.warning(self, "Error", "Valid regularization folder required when regularization is enabled")
                return
            # Collect selected regularization folders
            selected_reg_folders = []
            for folder_name, checkbox in self.reg_folder_checkboxes.items():
                if checkbox.isChecked():
                    selected_reg_folders.append(folder_name)
            if not selected_reg_folders:
                QtWidgets.QMessageBox.warning(self, "Error", "At least one regularization folder must be selected when regularization is enabled")
                return
            # Handle regularization seed
            if self.reg_seed_combo.currentText() == "Random":
                reg_seed = random.randint(0, 2147483647)
                self.reg_seed_spin.setValue(reg_seed)  # Update UI to show generated seed
                self.log_box.append(f"Generated regularization seed: {reg_seed}")
            else:  # Static mode
                reg_seed = self.reg_seed_spin.value()
                self.log_box.append(f"Using regularization seed: {reg_seed}")
        
        resize_enabled = self.resize_checkbox.isChecked()
        resize_pixels = self.resize_spin.value()
        self.worker = Worker(self.entries, dest, total, "total", convert_to_png=self.convert_to_png, 
                           regularization_folder=reg_folder, regularization_percent=reg_percent,
                           training_seed=training_seed, regularization_seed=reg_seed,
                           selected_reg_folders=selected_reg_folders,
                           kohya_project=True, training_mode=self.training_mode_combo.currentText(),
                           instance_prompt="", class_prompt="a person", repeats=3,
                           copy_training=self.copy_training_checkbox.isChecked(),
                           copy_regularization=self.copy_regularization_checkbox.isChecked(),
                           resize_enabled=resize_enabled, resize_pixels=resize_pixels)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.message.connect(self.log_box.append)
        self.worker.finished.connect(lambda: QtWidgets.QMessageBox.information(self, "Done", "Sampling complete"))
        self.worker.start()

    def _load_settings(self):
        dest = self.settings.value("dest", "")
        self.dest_edit.setText(dest)
        
        # Load last folder path
        self.last_folder_path = self.settings.value("last_folder_path", "")
        
        # Load unlock percentages checkbox state
        unlock_state = self.settings.value("unlock_percentages", False, type=bool)
        self.unlock_percentages_checkbox.setChecked(unlock_state)
        
        # Load regularization settings
        reg_enabled = self.settings.value("regularization_enabled", False, type=bool)
        self.regularization_checkbox.setChecked(reg_enabled)
        reg_folder = self.settings.value("regularization_folder", "")
        self.reg_folder_edit.setText(reg_folder)
        reg_percent = self.settings.value("regularization_percent", 150, type=int)
        self.reg_percent_spin.setValue(reg_percent)
        
        # Load total number of images
        total_images = self.settings.value("total_images", 1000, type=int)
        self.total_spin.setValue(total_images)
        
        # Load seed settings
        training_seed_mode = self.settings.value("training_seed_mode", "Random")
        self.training_seed_combo.setCurrentText(training_seed_mode)
        training_seed_value = self.settings.value("training_seed_value", random.randint(0, 999999), type=int)
        self.training_seed_spin.setValue(training_seed_value)
        
        reg_seed_mode = self.settings.value("reg_seed_mode", "Random")
        self.reg_seed_combo.setCurrentText(reg_seed_mode)
        reg_seed_value = self.settings.value("reg_seed_value", random.randint(0, 999999), type=int)
        self.reg_seed_spin.setValue(reg_seed_value)
        
        # Load training mode (training_mode_combo still exists)
        training_mode = self.settings.value("training_mode", "LoRA/DreamBooth")
        self.training_mode_combo.setCurrentText(training_mode)
        
        # Load copy checkboxes
        copy_training = self.settings.value("copy_training", True, type=bool)
        self.copy_training_checkbox.setChecked(copy_training)
        copy_regularization = self.settings.value("copy_regularization", True, type=bool)
        self.copy_regularization_checkbox.setChecked(copy_regularization)
        
        # Update regularization controls state
        self.toggle_regularization_controls()
        
        # Update UI based on training mode
        self.update_regularization_for_training_mode()
        
        folders_json = self.settings.value("folders", "[]")
        try:
            folders = json.loads(folders_json)
        except Exception:
            folders = []
        for data in folders:
            try:
                # Create entry with backward compatibility for missing fields
                # Filter data to only include fields that exist in the current FolderEntry
                import inspect
                entry_fields = set(inspect.signature(FolderEntry.__init__).parameters.keys()) - {'self'}
                filtered_data = {k: v for k, v in data.items() if k in entry_fields}
                entry = FolderEntry(**filtered_data)
            except Exception as e:
                # If there's still an error, create with just the path and defaults
                print(f"[DEBUG] Error loading folder entry: {e}, creating with defaults")
                entry = FolderEntry(path=data.get('path', ''))
            
            widget = FolderWidget(entry, remove_callback=self.remove_folder_widget, get_total_images=self.get_total_images, get_total_megapixels=self.get_total_megapixels, get_unlock_state=self.get_unlock_state)
            self.entries.append(entry)
            self.folder_widgets.append(widget)
            self.list_layout.addWidget(widget)
            widget.percent_spin.valueChanged.connect(self.update_total_percent_label)
            widget.percent_spin.valueChanged.connect(self.update_avg_mp_label)
            widget.percent_spin.valueChanged.connect(self.update_all_folder_mp_indicators)
            widget.percent_spin.valueChanged.connect(self.update_sampled_images_label)
        self.update_total_percent_label()
        self.update_all_folder_max_percents()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        self.update_sampled_images_label()
        # Ensure folder widgets' field visibility matches the loaded training mode
        self.update_regularization_for_training_mode()
        # Load LoRA regularization enabled state
        lora_reg_enabled = self.settings.value("lora_regularization_enabled", None)
        if lora_reg_enabled is not None:
            self.lora_regularization_enabled = lora_reg_enabled == 'true' or lora_reg_enabled is True

        # Force refresh by incrementing and decrementing total_spin value
        current_val = self.total_spin.value()
        self.total_spin.setValue(current_val + 1)
        self.total_spin.setValue(current_val)

        # In update_reg_summary, after updating the summary label, save checked folders
        checked_folders = [folder_name for folder_name, checkbox in self.reg_folder_checkboxes.items() if checkbox.isChecked()]
        self.settings.setValue("checked_reg_folders", checked_folders)

        # In _load_settings, after updating the regularization folders list, restore checked state
        checked_folders = self.settings.value("checked_reg_folders", [])
        # QSettings may return a string, a list, or QVariant
        if isinstance(checked_folders, str):
            import ast
            try:
                checked_folders = ast.literal_eval(checked_folders)
            except Exception:
                checked_folders = []
        if not isinstance(checked_folders, list):
            checked_folders = list(checked_folders) if checked_folders else []
        # If nothing saved, default to all checked
        if not checked_folders:
            for checkbox in self.reg_folder_checkboxes.values():
                checkbox.setChecked(True)
        else:
            for folder_name, checkbox in self.reg_folder_checkboxes.items():
                checkbox.setChecked(folder_name in checked_folders)

    def closeEvent(self, event):
        self.save_checked_reg_folders()
        self.settings.setValue("dest", self.dest_edit.text())
        self.settings.setValue("last_folder_path", self.last_folder_path)
        self.settings.setValue("unlock_percentages", self.unlock_percentages_checkbox.isChecked())
        self.settings.setValue("regularization_enabled", self.regularization_checkbox.isChecked())
        self.settings.setValue("regularization_folder", self.reg_folder_edit.text())
        self.settings.setValue("regularization_percent", self.reg_percent_spin.value())
        self.settings.setValue("total_images", self.total_spin.value())
        
        # Save seed settings
        self.settings.setValue("training_seed_mode", self.training_seed_combo.currentText())
        self.settings.setValue("training_seed_value", self.training_seed_spin.value())
        self.settings.setValue("reg_seed_mode", self.reg_seed_combo.currentText())
        self.settings.setValue("reg_seed_value", self.reg_seed_spin.value())
        
        # Save Kohya SS settings
        self.settings.setValue("training_mode", self.training_mode_combo.currentText())
        
        # Save copy checkboxes
        self.settings.setValue("copy_training", self.copy_training_checkbox.isChecked())
        self.settings.setValue("copy_regularization", self.copy_regularization_checkbox.isChecked())
        
        data = [e.__dict__ for e in self.entries]
        self.settings.setValue("folders", json.dumps(data))
        # Save LoRA regularization enabled state
        self.settings.setValue("lora_regularization_enabled", str(self.lora_regularization_enabled))
        super().closeEvent(event)

    def update_all_folder_max_percents(self):
        print(f"[DEBUG] Updating folder max percents, unlock state: {self.unlock_percentages_checkbox.isChecked()}")
        for w in self.folder_widgets:
            w.check_max_percent()
        # Force update of total percent label and MP indicators when unlock state changes
        self.update_total_percent_label()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        print(f"[DEBUG] Finished updating folder max percents")

    def update_all_folder_mp_indicators(self):
        total_images = self.get_total_images()
        
        # Calculate how many images will be sampled from each folder
        for i, w in enumerate(self.folder_widgets):
            w.update_image_count_and_megapixels()  # Ensure count is current
            
            if self.unlock_percentages_checkbox.isChecked():
                # In independent mode, calculate based on folder's own percentage
                images_to_sample = int(w.image_count * w.percent_spin.value() / 100)
            else:
                # In locked mode, calculate based on total images and folder percentage
                images_to_sample = int(total_images * w.percent_spin.value() / 100)
                # Clamp to available images in folder
                images_to_sample = min(images_to_sample, w.image_count)
            
            w.set_images_sampled_label(images_to_sample, w.image_count)
        
        # If percentages are unlocked, MP balancing is irrelevant
        if self.unlock_percentages_checkbox.isChecked():
            for w in self.folder_widgets:
                w.set_megapixel_used_label(0.0, 'gray')
            return
            
        # Update each folder's megapixel used label with new color coding
        # Estimate: for each folder, how many images will be sampled?
        folder_samples = [int(total_images * w.percent_spin.value() / 100) for w in self.folder_widgets]
        folder_avg_mp = [w.get_folder_megapixels() / w.image_count if w.image_count else 0 for w in self.folder_widgets]
        folder_est_mp_used = [folder_samples[i] * folder_avg_mp[i] for i in range(len(self.folder_widgets))]
        avg_mp_contrib = sum(folder_est_mp_used) / len(self.folder_widgets) if self.folder_widgets else 0
        for i, w in enumerate(self.folder_widgets):
            est_mp_used = folder_est_mp_used[i]
            if avg_mp_contrib == 0:
                color = 'black'
            else:
                diff_percent = abs(est_mp_used - avg_mp_contrib) / avg_mp_contrib * 100
                if diff_percent <= 20:
                    color = 'green'
                elif diff_percent <= 30:
                    color = 'orange'
                else:
                    color = 'red'
            w.set_megapixel_used_label(est_mp_used, color)
        self.update_avg_mp_label()

    def update_avg_mp_label(self):
        # If percentages are unlocked, MP balancing is irrelevant
        if self.unlock_percentages_checkbox.isChecked():
            self.avg_mp_label.setText("Average per-folder MP contribution: <b><span style='color:gray'>N/A (Independent Mode)</span></b>")
            return
            
        total_images = self.get_total_images()
        folder_samples = [int(total_images * w.percent_spin.value() / 100) for w in self.folder_widgets]
        folder_avg_mp = [w.get_folder_megapixels() / w.image_count if w.image_count else 0 for w in self.folder_widgets]
        folder_est_mp_used = [folder_samples[i] * folder_avg_mp[i] for i in range(len(self.folder_widgets))]
        total_est_mp = sum(folder_est_mp_used)
        avg_mp_contrib = total_est_mp / len(self.folder_widgets) if self.folder_widgets else 0
        avg_percent = 100 / len(self.folder_widgets) if self.folder_widgets else 0
        self.avg_mp_label.setText(f"Average per-folder MP contribution: <b>{avg_mp_contrib:.2f} MP</b> ({avg_percent:.1f}%)")

    def on_total_spin_changed(self):
        self.update_all_folder_max_percents()
        self.update_total_percent_label()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        self.update_available_images_label()

    def get_total_images(self):
        return self.total_spin.value()

    def get_total_megapixels(self):
        return sum(w.get_folder_megapixels() for w in self.folder_widgets)

    def get_unlock_state(self):
        return self.unlock_percentages_checkbox.isChecked()

    def update_available_images_label(self):
        """Update the label showing total available images from all folders"""
        total_available = 0
        for w in self.folder_widgets:
            w.update_image_count_and_megapixels()  # Ensure count is current
            total_available += w.image_count
        
        if total_available == 0:
            self.available_images_label.setText("(No images available)")
            self.available_images_label.setStyleSheet("color: gray;")
        else:
            target_total = self.total_spin.value()
            if target_total > total_available:
                color = "red"
                self.available_images_label.setText(f"(Available: {total_available} - Not enough!)")
            else:
                color = "green"
                self.available_images_label.setText(f"(Available: {total_available})")
            self.available_images_label.setStyleSheet(f"color: {color};")

    def adjust_all_folder_percents(self, delta):
        for w in self.folder_widgets:
            new_val = max(0, min(100, w.percent_spin.value() + delta))
            w.percent_spin.setValue(new_val)
        self.update_total_percent_label()
        self.update_all_folder_max_percents()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        self.update_available_images_label()

    def add_multiple_folders(self):
        # Pass the last used folder path to start from the same location
        dlg = MultiFolderSelectDialog(self, self.last_folder_path)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected_dirs = [d for d in dlg.selected_folders() if os.path.isdir(d)]
            folder_paths = [e.path for e in self.entries]
            added = 0
            for folder in selected_dirs:
                if folder not in folder_paths:
                    entry = FolderEntry(path=folder)
                    widget = FolderWidget(entry, remove_callback=self.remove_folder_widget, get_total_images=self.get_total_images, get_total_megapixels=self.get_total_megapixels, get_unlock_state=self.get_unlock_state)
                    self.entries.append(entry)
                    self.folder_widgets.append(widget)
                    self.list_layout.addWidget(widget)
                    widget.percent_spin.valueChanged.connect(self.update_total_percent_label)
                    widget.percent_spin.valueChanged.connect(self.update_avg_mp_label)
                    widget.percent_spin.valueChanged.connect(self.update_all_folder_mp_indicators)
                    widget.percent_spin.valueChanged.connect(self.update_sampled_images_label)
                    added += 1
                    self.last_folder_path = folder
                    self.settings.setValue("last_folder_path", folder)
            if added == 0 and selected_dirs:
                QtWidgets.QMessageBox.information(self, "No New Folders", "All selected folders were already added.")
            # --- Set percent so MP Utilized is green for all folders if possible ---
            if added > 0 and len(self.folder_widgets) > 0:
                for w in self.folder_widgets:
                    w.update_image_count_and_megapixels()
                N = len(self.folder_widgets)
                total_mp = sum(w.megapixels for w in self.folder_widgets)
                target_mp = total_mp / N if N else 0
                raw_percents = [(target_mp / w.megapixels) * 100 if w.megapixels > 0 else 0 for w in self.folder_widgets]
                percent_sum = sum(raw_percents)
                percents = [round(p * 100 / percent_sum) for p in raw_percents]
                diff = 100 - sum(percents)
                if percents:
                    percents[-1] += diff
                for w, perc in zip(self.folder_widgets, percents):
                    w.percent_spin.blockSignals(True)
                    w.percent_spin.setValue(max(0, min(100, perc)))
                    w.percent_spin.blockSignals(False)
                    self.update_total_percent_label()
        self.update_all_folder_max_percents()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        self.update_available_images_label()
        self.update_sampled_images_label()

    def refresh_directories(self):
        for w in self.folder_widgets:
            w._mp_cache_valid = False
            w.update_image_count_and_megapixels(force=True)
            w.update_caption_style_colors()
        self.update_total_percent_label()
        self.update_all_folder_max_percents()
        self.update_avg_mp_label()
        self.update_all_folder_mp_indicators()
        self.update_available_folders_list()

    def select_all_reg_folders(self):
        """Select all regularization folders"""
        for checkbox in self.reg_folder_checkboxes.values():
            checkbox.setChecked(True)
        self.update_reg_summary()

    def select_none_reg_folders(self):
        """Deselect all regularization folders"""
        for checkbox in self.reg_folder_checkboxes.values():
            checkbox.setChecked(False)
        self.update_reg_summary()

    def save_checked_reg_folders(self):
        """Save the currently checked regularization folders to QSettings"""
        checked_folders = [folder_name for folder_name, checkbox in self.reg_folder_checkboxes.items() if checkbox.isChecked()]
        self.settings.setValue("checked_reg_folders", checked_folders)

    def update_regularization_folders_list(self):
        """Update the list of regularization subfolders with checkboxes"""
        # Save checked state before clearing
        self.save_checked_reg_folders()
        # Clear existing checkboxes
        self.clear_regularization_folders_list()
        
        reg_folder = self.reg_folder_edit.text()
        if not reg_folder or not os.path.isdir(reg_folder):
            self.update_reg_summary()
            return
        
        # Identify all direct subfolders of the regularization folder
        subfolders = []
        for item in os.listdir(reg_folder):
            item_path = os.path.join(reg_folder, item)
            if os.path.isdir(item_path):
                # Count images in this subfolder (including nested subfolders)
                image_count = 0
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            image_count += 1
                
                if image_count > 0:  # Only include folders with images
                    subfolders.append((item, image_count))
        
        # If no subfolders with images, check if the root folder has images
        if not subfolders:
            root_image_count = 0
            for file in os.listdir(reg_folder):
                if os.path.isfile(os.path.join(reg_folder, file)):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        root_image_count += 1
            
            if root_image_count > 0:
                subfolders.append(("root", root_image_count))
        
        # Add checkboxes for each subfolder
        for folder_name, image_count in subfolders:
            display_name = f"{folder_name} ({image_count} images)"
            checkbox = QtWidgets.QCheckBox(display_name)
            checkbox.setChecked(True)  # Default to checked
            checkbox.stateChanged.connect(self.update_reg_summary)
            checkbox.stateChanged.connect(lambda _, fn=folder_name: self.save_checked_reg_folders())
            
            self.reg_folders_layout.addWidget(checkbox)
            self.reg_folder_checkboxes[folder_name] = checkbox
        
        self.update_reg_summary()

        # After adding all checkboxes in update_regularization_folders_list, restore checked state
        checked_folders = self.settings.value("checked_reg_folders", [])
        # QSettings may return a string, a list, or QVariant
        if isinstance(checked_folders, str):
            import ast
            try:
                checked_folders = ast.literal_eval(checked_folders)
            except Exception:
                checked_folders = []
        if not isinstance(checked_folders, list):
            checked_folders = list(checked_folders) if checked_folders else []
        # If nothing saved, default to all checked
        if not checked_folders:
            for checkbox in self.reg_folder_checkboxes.values():
                checkbox.setChecked(True)
        else:
            for folder_name, checkbox in self.reg_folder_checkboxes.items():
                checkbox.setChecked(folder_name in checked_folders)

    def clear_regularization_folders_list(self):
        """Clear the regularization folders list"""
        # Clear existing checkboxes
        for i in reversed(range(self.reg_folders_layout.count())):
            item = self.reg_folders_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        self.reg_folder_checkboxes.clear()
        self.update_reg_summary()

    def update_reg_summary(self):
        """Update the summary of selected regularization folders"""
        if not self.regularization_checkbox.isChecked():
            self.reg_summary_label.setText("")
            return
        
        total_images = 0
        selected_count = 0
        
        for folder_name, checkbox in self.reg_folder_checkboxes.items():
            if checkbox.isChecked():
                selected_count += 1
                
                reg_folder = self.reg_folder_edit.text()
                if folder_name == "root":
                    # Count images in root folder
                    for file in os.listdir(reg_folder):
                        if os.path.isfile(os.path.join(reg_folder, file)):
                            ext = os.path.splitext(file)[1].lower()
                            if ext in IMAGE_EXTENSIONS:
                                total_images += 1
                else:
                    # Count images in subfolder
                    subfolder_path = os.path.join(reg_folder, folder_name)
                    if os.path.isdir(subfolder_path):
                        for root, dirs, files in os.walk(subfolder_path):
                            for file in files:
                                ext = os.path.splitext(file)[1].lower()
                                if ext in IMAGE_EXTENSIONS:
                                    total_images += 1
        
        if selected_count == 0:
            self.reg_summary_label.setText("<span style='color:red'>Selected: 0 folders, 0 images</span>")
        else:
            # Calculate how many will be used
            main_dataset_size = self.total_spin.value()
            reg_percentage = self.reg_percent_spin.value()
            reg_images_to_use = int(main_dataset_size * reg_percentage / 100)
            
            if total_images >= reg_images_to_use:
                color = "green"
                self.reg_summary_label.setText(f"<span style='color:{color}'>Selected: {selected_count} folders, {total_images} images (Need: {reg_images_to_use})</span>")
            else:
                color = "red" 
                self.reg_summary_label.setText(f"<span style='color:{color}'>Selected: {selected_count} folders, {total_images} images (Need: {reg_images_to_use} - Not enough!)</span>")

    def update_sampled_images_label(self):
        # Calculate the total number of images to be sampled based on folder percentages and available images
        total = 0
        for w in self.folder_widgets:
            w.update_image_count_and_megapixels()
            # Use the same logic as in sampling: int(len(images) * percent / 100)
            count = int(w.image_count * w.percent_spin.value() / 100)
            total += count
        target = self.total_spin.value()
        if total == target:
            color = 'green'
        else:
            color = 'red'
        self.sampled_images_label.setText(f"<span style='color:{color}'>Sampled: {total} / {target}</span>")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Ensure progress bar matches log window width
        if hasattr(self, 'log_box') and hasattr(self, 'progress'):
            self.progress.setFixedWidth(self.log_box.width())

def convert_and_delete(src_path, png_path):
    from PIL import Image
    import os
    try:
        with Image.open(src_path) as im:
            if im.mode in ('RGBA', 'LA', 'P'):
                im.save(png_path, format="PNG", optimize=True)
            else:
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(png_path, format="PNG", optimize=True)
        os.remove(src_path)
        return True
    except Exception as e:
        if os.path.exists(png_path):
            os.remove(png_path)
        return str(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # Window size and position are set in MainWindow.__init__()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
