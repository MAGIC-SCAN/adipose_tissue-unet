# Environment Setup Guide

## Option 1: Conda Environment (Recommended)

### Create environment from YAML file
```bash
conda env create -f environment.yml
conda activate adipose-tf2
```

### Or create manually
```bash
conda create -n adipose-tf2 python=3.10
conda activate adipose-tf2
pip install -r requirements.txt
```

## Option 2: pip-only Installation

```bash
python -m venv adipose-tf2-env
source adipose-tf2-env/bin/activate  # On Windows: adipose-tf2-env\Scripts\activate
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import keras; print('Keras:', keras.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

Expected output:
```
TensorFlow: 2.13.1
Keras: 2.13.1
OpenCV: 4.8.0
NumPy: 1.23.5
```

## GPU Support

### For NVIDIA GPU (CUDA)
TensorFlow 2.13 requires:
- CUDA 11.8
- cuDNN 8.6

Install CUDA-enabled TensorFlow:
```bash
pip install tensorflow[and-cuda]==2.13.1
```

### Verify GPU availability
```bash
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## Troubleshooting

### Protobuf Version Conflicts
If you see protobuf errors, ensure version 3.20.3:
```bash
pip install --force-reinstall protobuf==3.20.3
```

### ONNX Installation Issues
ONNX is optional. If installation fails:
```bash
pip install --no-deps onnx==1.14.1
```

### OpenCV Import Errors
If `import cv2` fails, try:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.0.76
```

## Package Descriptions

| Package | Purpose |
|---------|---------|
| `tensorflow==2.13.1` | Deep learning framework (U-Net, InceptionV3) |
| `keras==2.13.1` | High-level neural network API |
| `opencv-python` | Image I/O and preprocessing |
| `scikit-image` | Image processing utilities |
| `numpy` | Numerical operations |
| `scipy` | Scientific computing (statistics) |
| `pandas` | Data manipulation and CSV handling |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical data visualization |
| `h5py` | HDF5 file support for model weights |
| `labelme` | JSON annotation format support |
| `tifffile` | TIFF image handling |
| `onnx` | Model export to ONNX format |
| `onnxruntime` | ONNX inference engine |
| `tf2onnx` | TensorFlow to ONNX converter |
| `scikit-learn` | Machine learning metrics |
| `tqdm` | Progress bars |
| `gdown` | Google Drive file downloads |
| `protobuf==3.20.3` | TF2.13 compatibility fix |

## Updating the Environment

### Update all packages
```bash
conda activate adipose-tf2
conda update --all
pip install --upgrade -r requirements.txt
```

### Update specific package
```bash
pip install --upgrade tensorflow==2.13.1
```

## Exporting Environment

### Export conda environment
```bash
conda env export > environment_frozen.yml
```

### Export pip requirements
```bash
pip freeze > requirements_frozen.txt
```
