# HummingbirdVision

A project developed for the Stoddard Lab to detect hummingbirds in video and image data in their HummerFlowerWatch project using vision models (YOLO, RTDETR) from Ultralytics. This was part of my junior paper for Princeton, developed in collaboration with Dr. Ben Hogan of the Stoddard Lab.


---

Table of contents:
1. [Install](#install)
2. [Setup](#setup)
3. [Usage](#usage)

---
## Install 

### Option A: Download ZIP  
1. Click **Code → Download ZIP** on GitHub.  
2. `cd` into the directory and follow [Setup](#setup) below

### Option B: Fork 
If you want to contribute changes, fork the repo and then propose them
1. Fork the repo on GitHub
2. Clone your fork
3. Add the upstream
   `git remote add upstream https://github.com/DavyBee/HummingbirdVision`
4. Follow [Setup](#setup) below
5. Make your changes, and push to your issue branch
6. Open a PR on GitHub against the main branch on this repo

---

## Setup

Follow one of the methods below to configure your Python environment.

### Option A: Conda Environment

1. **Update conda (or miniconda)**

   ```bash
   conda update conda
   ```
2. **Create the environment**

   ```bash
   conda env create -f environment.yml -n HFW-env
   ```
   * This creates a conda environment named `HFW-env`. For a custom name, change `HFW-env` into your own environment name. 
3. **Activate the environment**

   ```bash
   conda activate [env-name]
   ```

### Option B: venv Environment

1. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   ```
2. **Activate the environment**

   * **macOS / Linux**

     ```bash
     source venv/bin/activate
     ```
   * **Windows (PowerShell)**

     ```powershell
     .\venv\Scripts\Activate
     ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

**Make sure that your ipynb notebook kernel is made from the virtual environment interpreter!!**

### Predictions.ipynb:

<p>
Use this file to predict on your data. Place your data (either a video, multiple videos, or multiple individual frames) in the `data` folder. Then follow the instructions in the notebook to adjust the given adjustable parameters. There is a pipeline to detect with a You-Only-Look-Once model (YOLO) or a Real-Time Detection Transformer (RTDETR) model. The notebook gives indications on the pros and cons of each model, and how to adjust the parameters for different results. 
</p>


### Validation.ipynb:
<p>
Use this file to get metrics of your models against a labeled dataset. It is not an exact measure of the model's accuracy, as the src code performs postprocessing on the images (NMS). But it should be a good enough indication of accuracy.
</p>

### Training.ipynb
<p>
Use this file to fine-tune new models, either from the pretrained ultralytics models or from already fine-tuned models (such as the ones provided in the repository). Requires that you have a dataset in YOLO format.
</p>
---