# SkepticalNet: Safety-Aware Interactive Brain Lesion Segmentation (Inference Demo)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Weights License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Weights_License-CC_BY--NC--SA_4.0-yellow)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<br>

This repository contains the **inference code and pre-trained models** for the paper: *"SkepticalNet: A Safety-Aware Interactive Segmentation Framework for Neuro-Oncology"* (Accepted at MICCAI 2026).

Our method introduces a safety-aware training protocol that prevents "trigger-happy" hallucinations in interactive brain tumor segmentation. This demo showcases inference on a **Pre-treatment Adult Glioma** case from BraTS using point and bounding box interactions.


### ⚠️ License & Usage

To facilitate reproducibility while protecting the model's intellectual property:

* **Code:** Released under **Apache 2.0**.

* **Model Weights:** Released under **CC-BY-NC-SA 4.0** (Non-Commercial Use Only).

* **Base Framework:** Built upon [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive).


---


## 🚀 Quick Start: Google Colab (Recommended)

The easiest way to reproduce the Demo results (Dice scores + Visualization) without local installation.


1.  Download the `demo.ipynb` file from this repository.

2.  Upload it to [Google Colab](https://colab.research.google.com/).

3.  **Run All Cells.**

    * *Note:* The notebook automatically downloads the weights and sample data.

    * *Note:* You can also view the pre-computed outputs (Dice scores and plots) directly on GitHub by opening `demo.ipynb` above.


---


## 💻 Local Installation


If you prefer to run the demo locally, follow these steps.


### 1. Clone this repository

```bash
git clone https://github.com/stathisky-repo/SkepticalNet.git
cd SkepticalNet
```

### 2. Set up the environment

We recommend using a fresh Conda environment.

```bash
conda create -n skepticalNet python=3.10
conda activate skepticalNet
```

### 3. Install dependencies

This installs the official nnInteractive inference framework and other utilities.

```bash
pip install -r requirements.txt
```

### 4. Run the Demo

This script will automatically download the pre-trained "SkepticalNet" weights (~400MB) and a sample BraTS case, then perform inference on the sample data.

```bash
python demo.py
```


---



## 📝 Citation

If you find this code or our paper useful, please consider citing:

@inproceedings{skepticalnet2026,
  title={SkepticalNet: A Safety-Aware Interactive Segmentation Framework for Neuro-Oncology},
  author={Kyriazis, Efstathios and Kalliatakis, Grigorios and Bisdas, Sotirios and Tsiknakis, Manolis and Marias, Kostas},
  booktitle={Accepted at the 29th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026},
  note={To appear}
}
