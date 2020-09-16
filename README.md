# One shot GAN (OSGAN)

## 🛠 Installation & Set Up

1. Prerequisites

   ```sh
   Python >=3.7
   Tensorflow-gpu = 2.2
   ```

2. Install using requirements.txt

   ```sh
   pip install -r requirements.txt
   ```

3. Create Conda Environment

   ```sh
   conda create --name <env> --file requirements.txt
   ```

4. Alternative easy option

   ```sh
   Create Azure datascience VM
   Goto predefined tensorflow environment using
   conda activate py37_tensorflow
   ```


## 🚀 Building and Running for results

1. Goto desired dataset folder

   ```sh
   cd dataset
   ```

2. For OSGAN IID results

   ```sh
   python3 osgan_mnist_iid.py
   ```

3. For OSGAN Non-IID results (applicable only for image datsets)

   ```sh
   python3 osgan_mnist_non_iid.py
   ```

4. For Federated IID and NonIID results (can edit in code file for IID or Non-IID)

   ```sh
   python3 federated_dataset.py
   ```

## Folder structure

1. Plots

   ```sh
   Plots folder contains the generated plots for the paper (Results are taken from corresponding folder)
   ```

2. Dataset

   ```sh
   Each dataset has a corresponding folder, where results are divided based on clients, IID setup and the algorithm (OSGAN, Federated)
   ```

3.  Fashion MNIST

   ```sh
   For Fashion MNIST dataset we have two folders, where one folder contains the implementation of CGAN based OSGAN and other cantains GAN based OSGAN
   ```

4. Results

   ```sh
   Results folder in each setup wise results folder contains information regarding testing accuracy and training accuracies
   ```
