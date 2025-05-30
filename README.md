# Diffusion Training Dataset Composer

A powerful, user-friendly PyQt5-based tool for composing training datasets for diffusion models, supporting both LoRA/DreamBooth and Fine-tuning workflows. Designed for maximum flexibility, robust error handling, and a modern, intuitive UI.

---

## Features
- **Three-column UI**: Settings, source folders, and regularization controls for clear workflow.
- **LoRA/DreamBooth & Fine-tuning modes**: Mode-specific logic for blank captions, regularization, and folder handling.
- **Persistent settings**: Remembers last-used folders, checked states, and all user preferences across sessions.
- **Unique file handling**: Prevents overwrites by appending suffixes to duplicate filenames and their captions.
- **Color-coded folders**: Maximally distinct pastel palette for easy visual separation.
- **Regularization controls**: Select subfolders, set percentages, and see real-time image counts.
- **Seed controls**: Reproducible sampling for both training and regularization images.
- **Responsive layout**: Compact, modern, and resizable interface.
- **Cross-platform**: Works on Windows and Linux (see below).

---

## Installation

### Requirements
- Python 3.8+
- PyQt5
- Pillow

Install dependencies (recommended in a virtual environment):

```bash
pip install PyQt5 Pillow
```

---

## Usage

### Launching the Tool

#### Windows
Double-click or run:
```bash
python image_sampler_tool.py
```

#### Linux
Use the provided bash launcher script:
```bash
bash run_linux.sh
```
Or run directly:
```bash
python3 image_sampler_tool.py
```

If you get a Qt platform plugin error, try installing additional Qt dependencies for your distro (e.g., `sudo apt install python3-pyqt5` on Ubuntu/Debian).

---

## Step-by-Step Instructions

### 1. Project Setup (Left Column)
- **Training Mode**: Choose between `LoRA/DreamBooth` (for subject/character training) and `Fine-tuning` (for style/concept training). This affects available options and caption handling.
- **Destination Folder**: Set the output directory where your composed dataset will be saved. Use the `Browse` button to select a folder.

### 2. Training Data Setup
- **Total Images**: Set the total number of images to sample for your training set.
- **Resize to N Pixels**: Optionally resize images so their shortest side matches the specified pixel value.
- **Convert to PNG**: If checked, all non-PNG images will be converted to PNG in the output.
- **Copy Training Set**: If checked, training images will be copied to the output structure.
- **Copy Regularization Set**: If checked, regularization images will be copied as well (LoRA/DreamBooth mode only).
- **Seed Controls**: Set random/static seeds for reproducible sampling of both training and regularization images.
- **Unlock Folder Percentages**: Allows folder sampling percentages to be set independently (do not need to sum to 100%).

### 3. Source Folders (Middle Column)
- **Add Folder / Add Multiple**: Add one or more source folders containing your training images. Each folder gets its own color-coded section.
- **Folder %**: Set the percentage of the total images to sample from each folder.
- **Caption Percentages**: For each folder, set the percentage of images that should have blank, basic, detailed, or structured captions. (Blank only available in Fine-tuning mode.)
- **Instance/Class/Repeats**: (LoRA/DreamBooth mode) Set per-folder instance prompt, class prompt, and repeat count for Kohya SS structure.
- **Remove**: Remove a folder from the list.
- **Color Coding**: Each folder is visually separated for clarity.

### 4. Regularization Controls (Right Column, LoRA/DreamBooth mode only)
- **Regularization Folder**: Select the folder containing your regularization images.
- **Enable Regularization**: Toggle regularization image usage.
- **Regularization Percentage**: Set the percentage of regularization images to use (relative to total training images).
- **Subfolder Selection**: Select which subfolders (or root) to use for regularization images. Image counts are shown.
- **Select All/None**: Quickly select or deselect all subfolders.
- **Summary**: See how many images are available and how many are needed.

### 5. Execute (Bottom Left)
- **Start**: Begin the sampling and dataset composition process. Progress and log messages will appear.
- **Overwrite Warning**: If the destination contains existing `img` or `regularization` folders, you will be prompted to add to or clean them. No other folders are affected.
- **Log Window**: Shows progress, warnings, and errors during processing.

---

## Advanced Features & Tips
- **Persistent Settings**: The tool remembers your last-used folders, settings, and checked states between sessions.
- **Unique File Handling**: If duplicate filenames are encountered, the tool appends suffixes to avoid overwriting.
- **Caption Style Checking**: The UI color-codes caption style labels based on the presence of corresponding caption files.
- **Megapixel Balancing**: The tool provides feedback on the megapixel contribution of each folder to help balance your dataset.
- **Seed Reproducibility**: Use static seeds for exact reproducibility of sampled images.
- **Regularization in Fine-tuning**: Regularization controls are disabled in Fine-tuning mode.

---

## Troubleshooting
- **Qt Platform Plugin Error**: On Linux, install missing Qt dependencies (e.g., `sudo apt install python3-pyqt5`).
- **No Images Found**: Ensure your source folders contain supported image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.gif`).
- **Folder Already Exists Warning**: The tool will never delete or modify folders other than `img` and `regularization` in your destination.
- **Permissions**: Make sure you have write permissions to the destination folder.

---

## Contribution
Pull requests and issues are welcome! Please open an issue for bugs or feature requests.

## License
MIT License. See LICENSE file for details.

## Contact
For questions or support, open a GitHub issue or contact the maintainer.
