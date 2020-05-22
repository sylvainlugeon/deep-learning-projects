# Mini Project 2 
EE-559 - Deep Learning - EPFL - 2020

_**Benkley** Tyler_, _**Berdoz** Frédéric_, _**Lugeon** Sylvain_

## Context
The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.


## Content

* ```DL_MP2.ipynb```: Python notebook in which the project was developed. Contains all the figures (and more) presented in the report.

* ```test.py```: Python script that can be run without any arguments. It will train one model over 25 epoch using the deep learning framework developed in ```dl_lib``` (see ```ee559_miniprojects.pdf``` for more details). Its execution time is a couple of seconds on a dual core 2 GHz _Intel Core i5_.

* ```ee559_miniprojects.pdf```: Project definition.

* ```report.pdf```: The report of the project.

* ```figures``` folder: Contains figures presented in the report (and others).

* ```dl_lib``` folder: Contains the deep learning library developed in this project. Can be imported like a standard python library provided that this folder is next to the executable python script. It contains the following modules
    
    * ```__init__.py```: Constructor of the package.
    * ```activation.py```: Activation modules.
    * ```cell.py```: Parent (abstract) class for modules that must keep track of their internal state.
    * ```linear.py```: Linear layer module.
    * ```loss.py```: Loss modules
    * ```module.py```: Parent (abstract) class for all the modules representing layers of neural nets.
    * ```optimizer.py```: Optimizer modules.
    * ```sequential.py```: Sequential module (for multilayer perceptron).


## Prerequisite

This code was developed using ```python 3.7.3``` (with its standard libraries), and with ```torch.empty``` from ```pytorch 1.4.0```. In addition, for the visualization, ```numpy 1.16.4``` and ```matplotlib 3.1.3``` were used.





