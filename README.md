# SISA Unlearning Demo: Cat-Dog Classifier

    This repository contains a Jupyter Notebook demo showcasing SISA (Sharded, Isolated, Sliced, Aggregated) unlearning compared to Naive Unlearning on a cat-dog image classification task. The demo is part of my "Teach Me Something" series, illustrating efficient machine unlearning for AI privacy, inspired by Bourtoule et al. (2021).
    Watch the demo in action on YouTube: SISA vs. Naive Unlearning: Fast AI Privacy with Cat-Dog Classifier Demo 🎥
    Overview
    The demo trains a logistic regression classifier using features extracted from a pre-trained ResNet18 on a dataset of 100 training images (50 cats, 50 dogs) and 20 test images (10 cats, 10 dogs). It compares:

    Naive Unlearning: Retrains the entire model (99 images) after removing image #5.
    SISA Unlearning: Retrains only one shard (20 images) using 5 shards and 4 slices per shard, achieving faster unlearning with comparable accuracy.

# Key Results:

Naive Unlearning: ~0.001s, ~95% test accuracy.
SISA Unlearning: ~0.013s, ~75% test accuracy (ensemble trade-off).
SISA is faster, especially for larger datasets, and ensures privacy by efficiently removing data.

# Dataset
The demo uses the Kaggle Cat-Dog Classification dataset. The folder structure is:

    cat_dog_images/
    ├── train/
    │   ├── cat/
    │   │   ├── cat_1.jpg
    │   │   ├── ... (50 images)
    │   ├── dog/
    │   │   ├── dog_1.jpg
    │   │   ├── ... (50 images)
    ├── test/
    │   ├── cat/
    │   │   ├── cat_1.jpg
    │   │   ├── ... (10 images)
    │   ├── dog/
    │   │   ├── dog_1.jpg
    │   │   ├── ... (10 images)

Note: The Notebook uses cat/dog folder names, but ensure your dataset matches (some datasets use cats/dogs).
Prerequisites

Python 3.8+
Dependencies: Install via pip: pip install numpy scikit-learn torch torchvision Pillow


Hardware: GPU recommended for faster feature extraction (CPU works too).
Dataset: Download and place in /home/dhruv/Documents/AML_/cat_dog_images/ or update paths in the Notebook.

How to Run

Clone the Repository:
git clone https://github.com/your-username/sisa-unlearning-demo.git
cd sisa-unlearning-demo


# Set Up the Dataset:

Download the Kaggle dataset.
Extract and organize into the structure above.
Update train_dir and test_dir in the Notebook if your path differs from /home/dhruv/Documents/AML_/cat_dog_images/.


# Run the Notebook:

Open SISA_CODE_DEMO.ipynb in Jupyter:jupyter notebook SISA_CODE_DEMO.ipynb

Run cells sequentially to:
Load and preprocess images.
Extract ResNet18 features.
Train a baseline model.
Perform Naive and SISA unlearning (removing image #5).
Evaluate on test images with sample predictions.

Expected Output
Running the Notebook produces:

# Naive Unlearning Demo
Baseline training time: 0.003 seconds
Baseline test accuracy: 0.950
Removing image: .../cat/cat_5.jpg
Naive unlearning time (retrain all): 0.001 seconds
Naive unlearning test accuracy: 0.950

# SISA Unlearning Demo
SISA baseline test accuracy: 0.750
SISA unlearning time (retrain shard 0, slice 0+): 0.013 seconds
SISA unlearning test accuracy: 0.750

Sample predictions on test images ([Dog, Cat]):
Test images: ['.../test/cat/cat_1.jpg', '.../test/dog/dog_1.jpg']
Naive Unlearning: [1 0] (0=dog, 1=cat)
SISA Unlearning: [1 0] (0=dog, 1=cat)

# Notes

Folder Names: The Notebook assumes cat/dog subfolders. If your dataset uses cats/dogs, update the load_images function:folder = os.path.join(directory, "cat" if label == 1 else "dog")

Performance: SISA’s lower accuracy (75% vs. 95%) is due to ensemble trade-offs, as noted in Bourtoule et al. (2021). Timing differences are small here but scale with larger datasets.
GPU/CPU: The code auto-detects GPU availability for faster feature extraction.

# Resources

[1] An Introduction to Machine Unlearning(https://arxiv.org/pdf/2209.00939)
[2] Machine Unlearning: Solutions and Challenges(https://arxiv.org/pdf/2308.07061)
[3] A Survey of Machine Unlearning(https://arxiv.org/pdf/2209.02299)
[4] Machine Unlearning in 2024 - by  Ken Ziyu Liu(https://ai.stanford.edu/~kzliu/blog/unlearning)
[5] ARCANE: An Efficient Architecture for Exact Machine Unlearning(https://www.ijcai.org/proceedings/2022/0556.pdf)

Dataset: [Kaggle Cat-Dog Classification.](https://www.kaggle.com/datasets/dhruvpanchal1/cat-dog-classification)
Video: https://youtu.be/0W-w3Fh4VfQ
Code: SISA_CODE_DEMO.ipynb in this repository.

Contributing
Feel free to fork, submit issues, or contribute improvements! Ideas for enhancing the demo (e.g., new datasets, models) are welcome.

Contact
Questions? Reach out via GitHub Issues or comment on the YouTube video. Subscribe for more ML content in the "Teach Me Something" series! 🚀