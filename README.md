# Dataset Sampler

This tool provides a simple GUI for sampling images and captions from multiple folders.

## Features

- Add and remove multiple source folders
- Specify sampling percentage for each folder
- Configure caption style percentages (blank, basic, detailed, structured)
- Two sampling modes: Classic and Total Count
- Destination folder selection
- Progress bar and log output

The application is implemented with PyQt5 and stores the last used destination
folder via `QSettings`.

Run with:

```bash
python3 dataset_sampler.py
```

On Windows, use the provided batch script which sets up a virtual
environment and installs the required dependencies:

```bat
run_sampler.bat
```

Note: this is a minimal implementation and may not cover all desired features.
