spearmint: Bayesian optimization codebase
=========================================

Spearmint is a software package to perform Bayesian optimization. The Software is designed to automatically run experiments (thus the code name spearmint) in a manner that iteratively adjusts a number of parameters so as to minimize some objective in as few runs as possible.

## IMPORTANT: Please read about the license
Spearmint is under an **Academic and Non-Commercial Research Use License**.  Before using spearmint please be aware of the [license](LICENSE.md).  If you do not qualify to use spearmint you can ask to obtain a license as detailed in the [license](LICENSE.md) or you can use the older open source code version (which is somewhat outdated) at https://github.com/JasperSnoek/spearmint.  

## IMPORTANT: You are off the main branch!
This is the PESM branch. This branch contains the Predictive Entropy Search for Multiobjective Optimization. Predictive Entropy Search with Constraints should also work as described in the paper Predictive Entropy Search for Bayesian Optimization with Unknown Constraints. 

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

    Predictive Entropy Search for Multi-objective Bayesian Optimizaton
    Daniel Hernandez-Lobato, Jose Miguel Hernandez-Lobato, Amar Shah and Ryan Prescott Adams
    NIPS workshop on Bayesian optimization, 2015

This branch also includes the method in 

    Predictive Entropy Search for Bayesian Optimization with Unknown Constraints
    José Miguel Hernández-Lobato, Michael A. Gelbart, Matthew W. Hoffman, Ryan P. Adams, Zoubin Ghahramani
    arXiv: 1502.05312 [stat.ML], 2015.

    Predictive Entropy Search for Multi-objective Bayesian Optimizaton
    Daniel Hernandez-Lobato, Jose Miguel Hernandez-Lobato, Amar Shah and Ryan Prescott Adams
    NIPS workshop on Bayesian optimization, 2015

### STEP 1: Installation
1. Download/clone the spearmint code
2. Install the spearmint package using pip: "pip install -e \</path/to/spearmint/root\>" (the -e means changes will be reflected automatically)
3. Download and install MongoDB: https://www.mongodb.org/
4. Install the pymongo package using e.g., pip or anaconda
5. Install PyGMO package (this is used for solving inner multi-objective optimization problems with known, simple and fast objectives).
6. (Optional) Download and install NLopt: http://ab-initio.mit.edu/wiki/index.php/NLopt (see below for instructions)

### STEP 2: Setting up your experiment
1. Create a callable objective function. See ../examples/moo/branin.py as an example.
2. Create a config file. See ../examples/moo/config.json as an example. Here you will see that we specify the PESM acquisition function. Other alternatives are ParEGO, EHI, SMSego and SUR.

### STEP 3: Running spearmint
1. Start up a MongoDB daemon instance: mongod --fork --logpath \<path/to/logfile\> --dbpath \<path/to/dbfolder\>
2. Run spearmint: "python main.py \</path/to/experiment/directory\>"
(Try >>python main.py ../examples/toy)

### STEP 4: Looking at your results
Spearmint will output results to standard out / standard err and will also create output files in the experiment directory for each experiment. In addition, you can look at the results in the following ways:

1. The results are stored in the database. The program ../examples/moo/generate_hypervolumes.py extracts them from the database and computes some
perforamnce metrics, e.g., using the hypervolume.

### STEP 5: Cleanup
If you want to delete all data associated with an experiment (output files, plots, database entries), run "python cleanup.py \</path/to/experiment/directory\>"

#### (optional) Running multiple experiments at once
You can start multiple experiments at once using "python run_experiments.py \</path/to/experiment/directory\> N" where N is the number of experiments to run. You can clean them up at once with "python cleanup_experiments.py \</path/to/experiment/directory\> N". 

#### (optional) NLopt install instructions, without needing admin privileges 
1. wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
2. tar -zxvf nlopt-2.4.2.tar.gz
3. cd nlopt-2.4.2
4. mkdir build
5. ./configure PYTHON=PATH/TO/YOUR/PYTHON/python2.7 --enable-shared --prefix=PATH/TO/YOUR/NLOPT/nlopt-2.4.2/build/
6. make
7. make install
8. export PYTHONPATH=PATH/TO/YOUR/NLOPT/nlopt-2.4.2/build/lib/python2.7/site-packages/:$PYTHONPATH
9. (you can add line 8 to a .bashrc or equivalent file)
