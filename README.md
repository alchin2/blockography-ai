# Blockography.AI

Welcome to ACM AI's Fall 2025 Competition, **Blockography.AI**! In this competition, you will train your models to classify Minecraft biomes from real gameplay images. Compete against your peers to win Qualcomm Rubik Pi 3's, and get a chance to network with Qualcomm engineers!

## Table of Contents

- [Overview](#blockographyai)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
  - [Install Miniconda](#install-miniconda)
  - [Clone Repo and Create Environment](#clone-repo-and-create-environment)
- [Instruction](#instruction)
- [Submission and Evaluation](#submission-and-evaluation)
- [Competition Rules](#competition-rules)
- [FAQ](#faq)
- [Resources](#resources)

## Repo Structure

```text
BLOCKOGRAPHY-AI/
│
├── assets/                     # Images, reference files, or competition visuals
│
├── models/                     # Model training notebooks
│   ├── rf.ipynb                # Random Forest baseline
│   ├── ridge.ipynb             # Ridge Regression baseline
│   └── xgboost.ipynb           # XGBoost baseline
│
├── userkits/                   # Utility code and analysis tools
│   ├── EDA.ipynb               # Exploratory Data Analysis notebook
│   ├── features.py             # Feature extraction and transformation functions
│   └── utils.py                # Helper functions for preprocessing and evaluation
│
├── .gitignore                  # Files and directories to ignore in version control
├── environment.yml             # Conda environment setup file
├── requirements.txt            # Pip dependency list (optional)
└── README.md                   # Main documentation
```

## Installation

### Install Miniconda

First you need to install miniconda from <https://www.anaconda.com/download> to create environment. After logging in, choose "Miniconda Installers" and choose the one that matches your operating system (MacOS/Windows/Linux). Click the file you just download and follow the setup wizard. Remember to choose “Add Miniconda to my PATH environment variable.” Next, restart the terminal.

### Clone repo and create environment

To clone this repository, open your git bash or terminal and type in the following command:

```shell
git clone https://github.com/acmucsd/blockography-ai.git
cd blockography-ai
```

Create environment and install required package:

```shell
conda env create -f environment.yml
conda activate
```

⚠️ During the competition, if you want to install other packages, you can use either conda or pip to install. For example:

```shell
# use pip
pip install numpy
# or use conda
conda install numpy
```

## Instruction

In this competition, we expect you to mainly work on finding meaningful features to improve the performance of your model. We have provided a list of possible features in (`userkits/features.py`), and you should look over these features and understand why they might be helpful to classify minecraft biomes. You are also welcome to design your custom features.

We have provided with starter notebooks using classic machine learning models, such as random forest classifier, where you can change the features used in your model. You can also tune the hyperparameters of your model to achieve better accuracy.

In addition, you are welcome to use other models including deep learning models like convolutional neural network (CNN) and multi-layer perceptron (MLP), but we might not help you with the setup of dependencies and implementations.

## Submission and Evaluation

We split the dataset into two: train data and eval data. For the train data, you are given the features and the true labels (in `\train`). For the eval data you are only given the features but no labels. You are supposed to train your model using the train data and submit the prediction of the eval data.

To submit your prediction, run the final block (submission) in the starter notebooks. It will store the results of your prediction in the path specified (you can change this too). Next, go to <https://ai.acmucsd.com/portal> to manually upload the csv file. The submission portal will be closed at 6 pm.

On the website portal, you will see the public score of your submission, which is calculated using 50% of the test data. The final ranking will be based on the other 50% data (private score).

## Competition Rules

1. You are welcome to use any model for this competition as long as you can explain the algorithm/logic of your solution.
2. The use of LLM is allowed, but you have to make sure you review and test your code.
3. If we detect any behavior that intends to cheat (hacking, exploiting the evaluaion server, etc.), we have the right to cancel your eligibility.

## FAQ

**Q: Where can I see the current leaderboard?**

**A:** You can go to <https://ai.acmucsd.com/portal> to see your submissions and the current leaderboard.

**Q: What if I have questions related to the competition?**

**A:** Go to our discord server [ACM AI @ UCSD](https://acmurl.com/ai-discord) and find the channel for \<name\>. You can also directly reach out to our staff, but we may not help you with your solution.

## Resources

- [Feature Engineering for Image Classification](https://medium.com/data-science/feature-engineering-with-image-data-14fe4fcd4353) (you need to login to Medium)
- [ACM AI School #1 Slide](https://acmurl.com/ai-school-1-slides)
