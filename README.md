# ARISE2024
Resources for the ARISE program (summer 2024)

- Python tutorials
- Geometric, Kinematic and Dynamic Modeling (with 1R and 2R robots)
- Basic Control (Inverse Dynamics, PD + feedforward)

Notebooks are not fully Colab-compatible yet (some manual adjustments might be required for custom modules and figures import)

## A guide on how to setup your lab environment on Windows

### Install Anaconda

A virtual environment in Python is a isolated environment, separate from the main Python installation, that allows you to install packages and dependencies for a specific project without affecting the global Python installation. This helps to maintain a clean and organized environment, particularly when working with multiple projects that may have conflicting dependencies.

Anaconda is a distribution of the Python for scientific computing and data science. It includes pre-built packages for data analysis and visualization.

Anaconda includes tools, such as "conda", to create and manage virtual environments, allowing you to switch between environments and manage the packages installed in each environment.

1. Download the Anaconda installer for Windows from [here](https://www.anaconda.com/products/individual#windows).


2. Run the installer and follow the instructions. Make sure to select the option to add Anaconda to the PATH environment variable. Guide on how to add variables to the PATH environment variable can be found [here](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).

3. Open the Anaconda Navigator or open the Anaconda prompt by searching for "Anaconda Prompt" in the Start menu. There should be a (base) appearing to the left in your terminal. Close and restart your terminal if this did not work or try running ``` conda init ``` if that helps. 

### Setup the virtual environment

1. To create the virtual environment for the labs, change to your desired directory and clone Sebastien Kleff's repository found [here](https://github.com/skleff1994/ARISE2024_students) using

```
git clone https://github.com/skleff1994/ARISE2024_students
```

Note if you are unfamiliar with git, you can download the repository as a zip file by clicking on the green "Code" button on the top right of the repository page and selecting "Download ZIP" and unzip it.

2. Change directory to the laboratory folder:

```
cd ARISE2024_students
```

3. Create the virtual environment using conda:

```
conda env create -f environment.yml
```

This should create a virtual environment that has all the required packages we need for the laboratory.

4. Activate the virtual environment:

```
conda activate ARISE2024
```

ensure you have the environment activated by checking the name of the environment in the terminal. It should be (ARISE2024) instead of (base).

5. Run Jupyter Notebook after navigating to the folder you want (1-python, etc.):

```
jupyter notebook
```

This should open a new tab in your browser with the Jupyter Notebook interface. You can now open the notebooks.



