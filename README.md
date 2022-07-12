# PathMutualInformation
This repository contains Python code to calculate the path mutual information between two trajectories for our case studies. All the calculations in the paper "Dynamic Information Transfer in Stochastic Biochemical Networks" by Anne-Lena Moor and Christoph Zechner have been performed with this code. 

## Content
- `FF3.py` and `BS.py` contain the main codes to calculate the path mutual information for our case studies
- `FF2.py` contains code to calculate the path mutual information in the two node feed forward network 
- The files `analyticalfunctions.py`, `functions.py`, `trajectories.py`, `exactsolutions.py` and `BSfunctions.py` contain all the functions that are needed to use the main codes
- The folder `datafiles` contains the data generated to reproduce our figures and include the exemplary trajectory of the path mutual information and the numerically evaluated data points needed for figure 2 and 3

For further information, contact Anne-Lena Moor (moor@mpi-cbg.de). 

## Installation 
The provided code was written using Python v3.8.5 and uses the following libraries:
- [Numpy v1.19.2](https://www.numpy.org/)
- [Matplotlib v3.3.2](https://matplotlib.org/)
- [Scipy v1.5.2](https://scipy.org)


For installing Python and the required packages, one can use [Anaconda](https://www.anaconda.com/products/distribution#windows). Anaconda is a general package manager that contains all the for this code required packages. Instructions to install Python can be found [here](https://jupyter.readthedocs.io/en/latest/install.html). 

## Reproducing the figures of the paper 
The python scripts `FF3.py` and `BS.py` contain the source code to calculate the path mutual information for our case studies. The scripts are commented in order to give a better understanding for the pipeline. Executing the pipeline in the source codes will lead to a reproduction of the figures in the main text and appendix and provide an example of how to calculate the path mutual information with the presented methods. Before using the code, it is recommended to adapt the variables `MC` and `core` to the desired preciseness of the Monte Carlo average and the core of the used computer to allow parallelisation and to make calculations more efficient. The default sample size written in the code is `MC=10000` and the default number of cores is `core=1`. 

By using the script `FF3.py`, one can choose between a solution that also provides the exactly integrated filter equation by setting `exact=True` or by only using the moment closure method for the calculation by setting `exact=False` (default), which will lead to the figures presented in the main paper. If one decides to set `exact=True`, it is highly recommended to lower the sample size of the Monte Carlo average. The sample size we used for our calculations is `MC=600`. However, also a sample size of `MC=300` or `MC=100` might be sufficient for reproduction. 