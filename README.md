# Signature Presence Detection (Rule-Based)

This repository implements a lightweight, rule-based pipeline for detecting
whether a signature field contains handwritten content or is empty.

The approach is designed for production scenarios where the signature bounding
box is already provided by an upstream system.

## Problem Definition

Given a cropped signature field image, classify it as:

- **EMPTY**: no signature content present
- **CONTENT**: ink content present (signature or non-sign mark)

## Methodology

1. Grayscale normalization
2. Otsu-based binarization
3. Morphological noise removal
4. Connected Component Analysis (CCA)
5. Area-based heuristic decision

Key features:
- Total ink area
- Number of connected components
- Largest component area

No training is required.

## Supporting Open-Source References

- Signature Extraction Using Connected Component Labeling  
  https://github.com/PujithaGrandhi/Signature-extraction-using-connected-component-labeling

- Signature Detection from Images  
  https://github.com/meetamjadsaeed/Signature-Detection-from-Images

## Usage

Analyze a single image:

python poc.py path/to/image.png

Evaluate all images under the `data/` directory:

python poc.py

## Data

This repo does not ship large/private datasets. Put small demo images in
`data/samples/`.
