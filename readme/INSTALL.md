# Installation

The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.10.13, CUDA 12.1, and [PyTorch]((http://pytorch.org/)) 2.1.2.
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create -n sfcnet python=3.10.13
    ~~~
    And activate the environment.
    
    ~~~
    conda activate sfcnet
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    ~~~

3. Clone this repo:

    ~~~
    git clone git@github.com:anthonyweidai/SFCNet.git
    ~~~

4. Install the requirements

    ~~~
    cd $SFCNet_ROOT
    pip install -r requirements.txt
    ~~~
    
