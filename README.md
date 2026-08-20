# Image Processing Studio — VS Code Version

A Python desktop application for basic image processing, filtering, edge detection,
kernel visualization, grayscale conversion, pixel matrix analysis, and processed
image output.

## Technologies

- Python
- CustomTkinter
- OpenCV
- NumPy
- Pillow

## Features

1. Open JPG, JPEG, PNG, BMP and WEBP images.
2. Grayscale conversion.
3. Mean filter.
4. Gaussian filter.
5. Sobel X edge detection.
6. Sobel Y edge detection.
7. Sobel magnitude.
8. Laplacian edge detection.
9. Kernel visualization.
10. Pixel matrix analysis.
11. Side-by-side original and processed image preview.
12. Automatically saves the processed result as `output/Output.jpeg`.

## How to run in Visual Studio Code

### 1. Install Python

Install Python 3.10 or newer and make sure Python is available in the terminal.

### 2. Open this folder

Open the `Image_Processing_Studio_VSCode` folder in Visual Studio Code.

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the project

```bash
python main.py
```

The application window will open.

### 6. Generate the output

- Click **Open Image**.
- Select an image.
- Choose a filter.
- Click **Apply Filter**.
- The processed image appears in the right panel.
- The application automatically creates:

```text
output/Output.jpeg
```

You can also click **Save Output.jpeg**.

## Project structure

```text
Image_Processing_Studio_VSCode/
│
├── main.py
├── requirements.txt
├── README.md
└── output/
    └── Output.jpeg
```

## Important note

The VS Code application saves the output to a real image file. This is different
from displaying an output only inside a Jupyter Notebook cell.

If your original notebook uses additional processing steps that are not listed
in the project description, copy those steps into `main.py` as additional filters.
