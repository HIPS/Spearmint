Spearmint
=========================================

Spearmint is a software package to perform Bayesian optimization. The Software is designed to automatically run experiments (thus the code name spearmint) in a manner that iteratively adjusts a number of parameters so as to minimize some objective in as few runs as possible.

**IMPORTANT:** Spearmint is under an **Academic and Non-Commercial Research Use License**.  Before using spearmint please be aware of the [license](LICENSE.md).  If you do not qualify to use spearmint you can ask to obtain a license as detailed in the [license](LICENSE.md) or you can use the older open source code version (which is somewhat outdated) at https://github.com/JasperSnoek/spearmint.  

####Relevant Publications

Spearmint implements a combination of the algorithms detailed in the following publications:

    Practical Bayesian Optimization of Machine Learning Algorithms  
    Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams  
    Advances in Neural Information Processing Systems, 2012  

    Multi-Task Bayesian Optimization  
    Kevin Swersky, Jasper Snoek and Ryan Prescott Adams  
    Advances in Neural Information Processing Systems, 2013  

    Input Warping for Bayesian Optimization of Non-stationary Functions  
    Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams  
    International Conference on Machine Learning, 2014  

    Bayesian Optimization and Semiparametric Models with Applications to Assistive Technology  
    Jasper Snoek, PhD Thesis, University of Toronto, 2013  
  
    Bayesian Optimization with Unknown Constraints
    Michael Gelbart, Jasper Snoek and Ryan Prescott Adams
    Uncertainty in Artificial Intelligence, 2014

####Setting up Spearmint

**STEP 1: Installation**  

1. Install [python](https://www.python.org/), [numpy](http://www.numpy.org/), [scipy](http://www.numpy.org/), [pymongo](https://api.mongodb.org/python/current/). For academic users, the [anaconda](http://continuum.io/downloads) distribution is great. Use numpy 1.8 or higher. We use python 2.7.
2. Download/clone the spearmint code  
3. Install the spearmint package using pip: `pip install -e \</path/to/spearmint/root\>` (the -e means changes will be reflected automatically)  
4. Download and install MongoDB: https://www.mongodb.org/   
5. Install the pymongo package using e.g., pip `pip install pymongo` or anaconda `conda install pymongo`  

**STEP 2: Setting up your experiment**  
1. Create a callable objective function. See `./examples/simple/branin.py` as an example  
2. Create a config file. There are 3 example config files in the ../examples directory. Note 1: There are more parameters that can be set in the config files than what is shown in the examples, but these parameters all have default values. Note 2: By default Spearmint assumes your function is noisy (non-deterministic). If it is noise-free, you should set this explicitly as in the ../examples/simple/config.json file.

**STEP 3: Running spearmint**  
1. Start up a MongoDB daemon instance:  
`mongod --fork --logpath <path/to/logfile\> --dbpath <path/to/dbfolder\>`  
2. Run spearmint: `python main.py \</path/to/experiment/directory\>`

**STEP 4: Looking at your results**  
Spearmint will output results to standard out / standard err. You can also load the results from the database and manipulate them directly. 
