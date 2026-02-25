# Installation & Setting Up the Environment

* For your convenience, we provide an `environemnt.yml` file which installs the required packages in a `conda`
  environment named `dlp`. Alternatively, you can use `pip` to install `requirements.txt`.
    * Use the terminal or an Anaconda Prompt and run the following command `conda env create -f environment.yml`.

If you prefer to set-up an environment manually, we provide the steps required set it up. 
We assume Anaconda or Miniconda is installed for environment management.

1. Create a new environment: `conda create -n dlp python=3.10`
2. Install PyTorch and CUDA (the command may vary depending on your system, change appropriately. https://pytorch.org/get-started/locally/):

    `pip3 install torch torchvision`

3. Run the following commands to install remaining `conda` libraries (can also use `pip` if you prefer):

    `conda install -c conda-forge numpy` (should already be installed from (1))
    `conda install -c conda-forge matplotlib`
    `conda install -c conda-forge tqdm`
    `conda install -c conda-forge scipy`
    `conda install -c conda-forge scikit-image`
    `conda install -c conda-forge imageio`
    `conda install -c conda-forge h5py`
    `conda install -c conda-forge notebook` (if you want to be able to run Jupyter Notebooks)
    `conda update ffmpeg`

4. Install `pip` packages:

    `pip install opencv-python`
    `pip install accelerate`
    `pip install piqa`
    `pip install einops`

5. (OPTIONAL) Clean `conda` cache: `conda clean --all`
