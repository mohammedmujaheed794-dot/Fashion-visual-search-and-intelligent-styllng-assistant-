# Fashion Visual Search & Intelligent Styling Assistant

## Project Overview
This project performs image-based fashion search and provides intelligent styling recommendations using Deep Learning (ResNet50) and Computer Vision techniques.

## ML Pipeline Flow

1.  **Data Loading & Validation** (`src/data_loader.py`)
    *   Traverses local dataset directories.
    *   Infers categories from folder names.
    *   Validates image integrity and formats.

2.  **Preprocessing & Feature Extraction** (`src/feature_extractor.py`)
    *   **Model**: ResNet50 (Pretrained on ImageNet).
    *   **Input**: Resizes images to (224, 224, 3).
    *   **Output**: 2048-dimensional dense feature vectors.
    *   **Normalization**: L2 Normalization applied for cosine similarity compatibility.

3.  **Embedding Storage**
    *   Embeddings saved as `.npy` file.
    *   Metadata (paths, categories) saved as `.pkl`.

4.  **Visual Similarity Search** (`src/search_engine.py`)
    *   Uses Cosine Similarity (via `sklearn.neighbors.NearestNeighbors`).
    *   Retrieves Top-K most visually similar images.

5.  **Intelligent Styling Assistant** (`src/stylist.py`)
    *   Identifies input item's category.
    *   Applys compatibility rules (e.g., Top -> Bottom + Shoes).
    *   Suggests items from complementary categories.

## Setup & Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Initialize & Build Index
Run this command to scan the dataset and generate embeddings.
```bash
python main.py init --dataset "path/to/dataset"
```

### 2. Visual Search
Search for items similar to a query image.
```bash
python main.py search path/to/query_image.jpg
```

### 3. Get Outfit Recommendations
Get styling suggestions for an item.
```bash
python main.py style path/to/my_shirt.jpg
```

## Directory Structure
```
.
├── main.py                 # CLI Entry point
├── requirements.txt        # Dependencies
├── src/
│   ├── config.py           # Configuration
│   ├── data_loader.py      # Dataset handling
│   ├── feature_extractor.py# CNN Model & Extraction
│   ├── search_engine.py    # Similarity logic
│   └── stylist.py          # Recommendation logic
└── artifacts/              # Generated embeddings & metadata
```

## Technologies
- **TensorFlow/Keras**: ResNet50 model.
- **Scikit-Learn**: Nearest Neighbors search.
- **OpenCV/PIL**: Image processing.
- **NumPy**: Matrix operations.
