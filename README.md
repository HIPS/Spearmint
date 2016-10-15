spearmint: Bayesian optimization codebase
=========================================

Spearmint is a software package to perform Bayesian optimization. The Software is designed to automatically run experiments (thus the code name spearmint) in a manner that iteratively adjusts a number of parameters so as to minimize some objective in as few runs as possible.

## IMPORTANT: Please read about the license
Spearmint is under an **Academic and Non-Commercial Research Use License**.  Before using spearmint please be aware of the [license](LICENSE.md).  If you do not qualify to use Spearmint you can ask to obtain a license as detailed in the [license](LICENSE.md) or you can use the older open source code version (which is somewhat outdated) at https://github.com/JasperSnoek/spearmint.  

## IMPORTANT: You are off the main branch!
This is the PESC branch. This branch contains the Predictive Entropy Search with Constraints (PESC) acquisition function, described in the paper Predictive Entropy Search for Bayesian Optimization with Unknown Constraints (http://arxiv.org/abs/1502.05312). Note: using PESC <i>without</i> constraints results in a method that is similar (but not exactly equivalent) to Predictive Entropy Search (http://arxiv.org/abs/1406.2541). This branch also comes with a basic 2D plotting routine. 

Update: as of 23-08-2016 this branch also includes support for decoupling as well as PESC-F as discussed in the paper A General Framework for Constrained Bayesian Optimization using Information-based Search.

## Relevant Publications

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

This branch also includes the methods in 

    Predictive Entropy Search for Bayesian Optimization with Unknown Constraints
    José Miguel Hernández-Lobato, Michael A. Gelbart, Matthew W. Hoffman, Ryan P. Adams, Zoubin Ghahramani
    International Conference on Machine Learning, 2015.
    
    Constrained Bayesian Optimization and Applications
    Michael A. Gelbart, PhD Thesis, Harvard University, 2015
    
    A General Framework for Constrained Bayesian Optimization using Information-based Search
    José Miguel Hernández-Lobato, Michael A. Gelbart,  Ryan P. Adams, Matthew W. Hoffman, Zoubin Ghahramani
    Journal of Machine Learning Research, 2016.
    
## Usage instructions
    
### STEP 1: Installation
1. Download/clone the spearmint code
2. Install the spearmint package using pip: "pip install -e \</path/to/spearmint/root\>" (the -e means changes will be reflected automatically)
3. Download and install MongoDB: https://www.mongodb.org/
4. Install the pymongo package using e.g., pip or anaconda
5. (Optional) Download and install NLopt: http://ab-initio.mit.edu/wiki/index.php/NLopt (see below for instructions)

### STEP 2: Setting up your experiment
1. Create a callable objective function. See ../examples/toy/toy.py as an example.
2. Create a config file. See ../examples/toy/config.json as an example. Here you will see that we specify the PESC acquisition function rather than the default, which is Expected Improvement (EI).

### STEP 3: Running spearmint
1. Start up a MongoDB daemon instance: `mongod --fork --logpath \<path/to/logfile\> --dbpath \<path/to/dbfolder\>`
2. Run spearmint: `python main.py \</path/to/experiment/directory\>`
(Try >>python main.py ../examples/toy)

### STEP 4: Looking at your results
Spearmint will output results to standard out / standard err and will also create output files in the experiment directory for each experiment. In addition, you can look at the results in the following ways:

1. To print all results, run `python print_all_results.py \</path/to/experiment/directory\>`
2. To create a plot showing the objective function decreasing over time, go to the `visualizations` directory and run `python progress_curve.py \</path/to/experiment/directory\>`. The result will appear in a "plots" subdirectory of the experiment directory. If you look at the bottom of the progress curve file you will see a number of options such as plotting in the log scale, etc. 
3. (2D objective functions only) To create plots of a 2D objective function (such as the examples provided), go to the visualizations directory and run `python plots_2d.py \</path/to/experiment/directory\>`. They will appear in a `plots` subdirectory in the experiment directory.

### STEP 5: Cleanup
If you want to delete all data associated with an experiment (output files, plots, database entries), run `python cleanup.py \</path/to/experiment/directory\>`

#### (optional) Running multiple experiments at once
You can start multiple experiments at once using `python run_experiments.py \</path/to/experiment/directory\> N` where N is the number of experiments to run. You can clean them up at once with "python cleanup_experiments.py \</path/to/experiment/directory\> N". Some of the helper functions above, such as progress_curve.py, are designed to work with this paradigm, so for example you can do `python progress_curve.py \</path/to/experiment/directory\> --repeat=10` to plot the average of 10 experiments with error bars.

#### (optional) NLopt install instructions, without needing admin privileges 
1. `wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz`
2. `tar -zxvf nlopt-2.4.2.tar.gz`
3. `cd nlopt-2.4.2`
4. `mkdir build`
5. `./configure PYTHON=PATH/TO/YOUR/PYTHON/python2.7 --enable-shared --prefix=PATH/TO/YOUR/NLOPT/nlopt-2.4.2/build/`
6. `make`
7. `make install`
8. `export PYTHONPATH=PATH/TO/YOUR/NLOPT/nlopt-2.4.2/build/lib/python2.7/site-packages/:$PYTHONPATH`
9. (you can add line 8 to a .bashrc or equivalent file)
 
