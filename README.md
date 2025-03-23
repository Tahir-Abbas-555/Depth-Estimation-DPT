# Depth Estimation using DPT

[Live Demo](https://huggingface.co/spaces/Tahir5/Depth-Estimation-DPT)

This Streamlit-based application estimates the depth map of an uploaded image using the Intel DPT (Dense Prediction Transformer) model.

## Features
- Upload an image in JPG, PNG, or JPEG format.
- Predict depth maps using the pre-trained **DPT-Large** model from Hugging Face.
- Visualize the original image and its corresponding depth map.

## Installation

### Prerequisites
Ensure you have Python installed (>=3.7). Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Dependencies
The required Python packages are:
```txt
streamlit
matplotlib
torch
numpy
transformers
Pillow
requests
```

## Usage
Run the Streamlit app with:

```bash
streamlit run app.py
```

Then, upload an image and view its predicted depth map.

## Project Structure
```
.
├── app.py              # Main application file
├── requirements.txt    # List of dependencies
├── README.md           # Documentation
```

## Model Details
This application uses **DPT-Large**, a transformer-based model for depth estimation, available on Hugging Face under the **Intel/dpt-large** repository.

## Example Output
After uploading an image, the app displays:
1. The original image.
2. The depth map visualization using the inferno colormap.

## License
This project is licensed under the MIT License.

## Author
[Tahir Abbas Shaikh](https://github.com/Tahir-Abbas-555)

