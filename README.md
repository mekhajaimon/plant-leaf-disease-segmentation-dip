# Plant Leaf Disease Region Segmentation and Severity Analysis Using Digital Image Processing

## About the Project

This project uses Digital Image Processing techniques to identify and segment disease-affected regions in plant leaf images.

The project focuses on **disease region segmentation**, not disease classification. Different image processing techniques are applied and compared to find which method performs better in identifying the affected regions.

The affected area is also estimated to provide a simple severity indication.

## Objectives

- To preprocess plant leaf images using grayscale conversion and Gaussian Blur.
- To segment disease-affected regions using different image processing techniques.
- To apply Otsu Thresholding, Adaptive Thresholding, and Canny Edge Detection.
- To improve segmentation using morphological operations.
- To compare the segmentation methods using IoU, Dice Coefficient, and Accuracy.
- To estimate the affected area of a leaf.
- To provide a simple severity level based on the estimated affected area.
- To analyze the limitations of the segmentation results.

## Dataset

The project uses a plant leaf disease segmentation dataset containing leaf images and their corresponding ground truth masks.

- Images: 2,940
- Masks: 2,940
- Image type: Colour images
- Mask type: Grayscale masks
- Ground truth masks: Used for evaluating segmentation results

## Methodology

The main workflow of the project is:

1. Dataset extraction and loading
2. Grayscale conversion
3. Gaussian Blur preprocessing
4. Otsu Thresholding
5. Morphological Opening and Closing
6. Adaptive Thresholding
7. Canny Edge Detection
8. Evaluation using IoU, Dice Coefficient, and Accuracy
9. Selection of the best-performing method
10. Affected area estimation
11. Severity estimation
12. Visualisation of the affected region

## Techniques Used

### Otsu Thresholding

Otsu Thresholding automatically selects a threshold value based on pixel intensity and creates a binary segmentation mask.

### Morphological Processing

Morphological Opening and Closing are used to remove small unwanted regions and fill small gaps in the segmentation mask.

### Adaptive Thresholding

Adaptive Thresholding calculates threshold values for local regions of the image. It is useful when intensity or lighting is not uniform.

### Canny Edge Detection

Canny Edge Detection identifies important boundaries in the image. The detected edges are converted into a filled mask for comparison with the other segmentation methods.

## Results

The three segmentation methods were evaluated using Average IoU on 50 leaf images.

| Method | Average IoU |
|---|---:|
| Otsu Thresholding | 0.0711 |
| Adaptive Thresholding | **0.1547** |
| Canny Edge Detection | 0.1419 |

Adaptive Thresholding achieved the highest Average IoU and was selected as the final segmentation method.

Further evaluation of Adaptive Thresholding gave:

- Average Dice Coefficient: **0.2565**
- Average Accuracy: **0.7388**

For the sample image used for severity analysis:

- Ground Truth Affected Area: **6.86%**
- Predicted Affected Area: **33.57%**
- Estimated Severity: **High**

The difference between the ground truth and predicted area shows that the method can over-segment some healthy regions.

## Limitations

- Adaptive Thresholding can sometimes identify healthy regions as diseased.
- The predicted affected area may be higher than the actual affected area.
- Performance can change with lighting, background, leaf colour, and disease patterns.
- The severity calculation is based only on the percentage of pixels in the predicted mask.
- The project does not classify the specific type of plant disease.
- Basic image processing methods may not perform equally well on all leaf images.

## Future Scope

- Use advanced segmentation techniques to improve accuracy.
- Test the system with a larger and more diverse dataset.
- Add image enhancement techniques for different lighting and backgrounds.
- Improve severity estimation using more reliable measurements.
- Extend the system to identify different plant diseases using machine learning or deep learning.
- Develop a web or mobile application for uploading leaf images and viewing the detected affected regions.

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Google Colab

## Repository Contents

- `DIP_MINI (3).ipynb` — Google Colab project notebook
- `source_code.py` — Python source code
- `requirements.txt` — Required Python libraries
- `screenshots/` — Project screenshots and results
- `report/` — Project report PDF

## GitHub Repository

[View the complete project on GitHub](https://github.com/mekhajaimon/plant-leaf-disease-segmentation-dip)
