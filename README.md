# Blockography.AI

![Logo](assets/logo.png)

Welcome to ACM AI's Fall 2025 Competition, **Blockography.AI**! In this competition, you will train your models to classify Minecraft biomes from real gameplay images. Compete against your peers to win Qualcomm Rubik Pi 3's, and get a chance to network with Qualcomm engineers!

> [!IMPORTANT]
> **How to get help**
>
> There are a few ways to get help, so you can pick the way that works best for you:
> - Discord: You can reach us on Discord in the #ai channel.
> - In person: You can find us in the Fishbowl (room B225 in the basement of the CSE building). We'll also be periodically walking around.

## Competition Timeline

- 10 am - 11 am: check-in
- 12 pm or 1 pm: lunch
- 5 pm: Submission ends
- 5 pm - 6 pm Qualcomm social time
- 6 pm: announce winners

## Table of Contents

- [Overview](#blockographyai)
- [Competition Timeline](#competition-timeline)
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
blockography-ai/
│
├── assets/                     # logo
│
├── outputs/                    # Storing outputs for your models
│   └── sample_submission.csv
│
├── models/                     # Model training notebooks
│   ├── rf.ipynb                # Random Forest baseline
│   ├── ridge.ipynb             # Ridge Regression baseline
│   └── xgboost.ipynb           # XGBoost baseline
│
├── userkits/                   # Utility code and analysis tools
│   ├── EDA.ipynb               # Exploratory Data Analysis notebook
│   ├── features.py             # Feature extraction and transformation functions
│   ├── utils.py                # Helper functions for preprocessing and evaluation
│
├── train_data/                 # Training data (download from Google Drive)
│   ├── bamboo_jungle/          # Example biome folder containing several images
│   │   ├── <file_name>.jpg     # Image files for training
│   │   └── ...
│   └── ...                     # Other biome folders with several images each
│
├── eval_data/                  # Evaluation data (download from Google Drive)
│   ├── <file_name>.jpg         # Several image files for evaluation
│   └── ...
│
├── .gitignore                  # Files and directories to ignore in version control
├── environment.yml             # Conda environment setup file
├── requirements.txt            # Pip dependency list (optional)
└── README.md                   # Main documentation
```

## Installation

In this section, you'll install Git, Miniconda, and a code editor. If you already have these tools installed, you can skip to the [Clone Repo and Create Environment](#clone-repo-and-create-environment) section.

### Prerequsites

Please make sure you've installed the followign tools:

- **Git**, which should already be installed on your system if you use MacOS or Linux. 
If you're on Windows, you can download it from <https://git-scm.com/download/win>, or use Windows Subsystem for Linux (WSL).
- A code editor. We strongly recommend using [**Visual Studio Code**](https://code.visualstudio.com/download), but you can also use other code editors.
- **Miniconda**, which is a package that will install Python and other tools that you will need for this competition.

### Install Miniconda

First you need to install miniconda from <https://www.anaconda.com/download> to create environment. After logging in, choose "Miniconda Installers" and choose the one that matches your operating system (MacOS/Windows/Linux). Click the file you just download and follow the setup wizard. Remember to choose “Add Miniconda to my PATH environment variable.” Next, restart the terminal.

*Note:* you may need to run `conda init` to finish setup.

### Clone repo and create environment

To clone this repository, open your git bash or terminal and type in the following command:

```shell
git clone https://github.com/acmucsd/blockography-ai.git
cd blockography-ai
```

Create environment and install required package:

```shell
conda env create -f environment.yml
conda activate ai-comp-env
```

⚠️ During the competition, if you want to install other packages, you can use either conda or pip to install. For example:

```shell
# use pip
pip install numpy
# or use conda
conda install numpy
```

### VS Code Setup

After using VS Code to open the repository, open `rf.ipynb`, click "Select Kernel" on the top right. If your VS Code doesn't have Python and Jupyter extensions, click "Install/Enable suggested extensions." After the extensions are installed, click "Select Kernel" again, and choose "Python Environments" and then "ai-comp-dev" on the top command bar. After the kernel is set, run the first code block. If it runs successfully, you are good to go!

### Download Data

You need to download data from Google Drive here:

- train_data: <https://drive.google.com/drive/folders/12ZZKnWFAvVhDVx37DvqizEtQ19LURluD?usp=drive_link>
- eval_data: <https://drive.google.com/drive/folders/1UAuCqjCoODF6vPLM80FFADENi17nYzre?usp=drive_link>

You then need to unzip the download zip file and extract it in the root of this directory (e.g. `/Users/<username>/blockography-ai` in MacOS). Make sure `train_data` and `eval_data` are stored in the root level or you'll have path issue when you run the notebooks.

## Instructions

In this competition, we expect you to mainly work on finding meaningful features to improve the performance of your model. We have provided a list of possible features in (`userkits/features.py`), and you should look over these features and understand why they might be helpful to classify minecraft biomes. You are also welcome to design your custom features.

We have provided with starter notebooks using classic machine learning models, such as random forest classifier, where you can change the features used in your model. You can also tune the hyperparameters of your model to achieve better accuracy.

### Deep Learning with PyTorch

While we primarily encourage you to focus on feature engineering and 'classical' machine learning models, we understand that deep learning models can also be used to solve this problem. You're welcome to use other models including deep learning models like convolutional neural networks (CNN) and multi-layer perceptrons (MLP). 

If you'd like to use deep learning models, we recommend using PyTorch, as we've provided a dataset and starter code. 

**Few things to note:**

- We've provided a PyTorch dataset, which you can import as such:

  ```python
  from userkits.torch_dataset import MinecraftTorchDataset
  ```

- To help with this, we've also provided a rough "scaffold" notebook that you can use as a starting point (see `userkits/pytorch.ipynb`)
- However, unlike the starter notebooks for classical machine learning, you'll need to implement the model yourself. 
- We may not be able to help you with the setup of dependencies and implementations.

## Submission and Evaluation

We split the dataset into two: train data and eval data. For the train data, you are given the features and the true labels (in `\train_data`). For the eval data (in `\eval_data`) you are only given the features but no labels. You are supposed to train your model using the train data and submit the prediction of the eval data.

To submit your prediction, run the final block (submission) in the starter notebooks. It will store the results of your prediction in a CSV file at the path specified (you can change this too). Next, go to <https://ai.acmucsd.com/portal> to manually upload the CSV file. **Please note that the submission portal will be closed at 5 pm.**

On the website portal, you will see the public score of your submission, which is calculated using 50% of the test data. The final ranking will be based on the other 50% data (private score).

## Competition Rules

1. You are welcome to use any model for this competition as long as you can explain the algorithm or logic behind your solution.
2. Winners will be interviewed to discuss their approach and solution before prizes are awarded.
3. The use of LLM is permitted, but participants are responsible for reviewing and testing all submitted code.
4. Any attempt to cheat, including hacking, exploiting the evaluation server, or other dishonest behavior, will result in disqualification.

## FAQ

**Q: Where can I see the current leaderboard?**

**A:** You can go to <https://ai.acmucsd.com/portal> to see your submissions and the current leaderboard.

**Q: What if I have questions related to the competition?**

**A:** Go to our discord server [ACM AI @ UCSD](https://acmurl.com/ai-discord) and find the channel for blockagraphy.ai/q-and-a. You can also directly reach out to our staff, but we may not help you with your solution.

## Resources

- [Feature Engineering for Image Classification](https://medium.com/data-science/feature-engineering-with-image-data-14fe4fcd4353) (you need to login to Medium)
- [ACM AI School #1 Slide](https://acmurl.com/ai-school-1-slides)
