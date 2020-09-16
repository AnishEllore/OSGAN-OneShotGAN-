# One shot GAN (OSGAN)

## 🛠 Installation & Set Up

1. Prerequisites

   ```sh
   Python >=3.7
   Tensorflow-gpu = 2.2
   ```

2.1 Install using requirements.txt

   ```sh
   pip install -r requirements.txt
   ```
   
2.2 Create Conda Environment

   ```sh
   conda create --name <env> --file requirements.txt
   ```
   
3. Alternative easy option

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

2.1 For OSGAN IID results

   ```sh
   python3 osgan_mnist_iid.py
   ```

2.2 For OSGAN Non-IID results (applicable only for image datsets)

   ```sh
   python3 osgan_mnist_non_iid.py
   ```

3 For Federated IID and NonIID results (can edit in code file for IID or Non-IID)

   ```sh
   python3 federated_dataset.py
   ```

## Plots:

Plots folder contains the generated plots for the paper (Results are taken from corresponding folder)
<br/>

---

## Dataset:

Each dataset has a corresponding folder, where results are divided based on clients, IID setup and the algorithm (OSGAN, Federated)
<br/>

#### Results:

Results folder in each setup wise results folder contains information regarding testing accuracy and training accuracies

---

### Fashion MNIST:

For Fashion MNIST dataset we have two folders, where one folder contains the implementation of CGAN based OSGAN and other cantains GAN based OSGAN
