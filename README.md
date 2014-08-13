spearmint: Bayesian optimization codebase
=========================================

Spearmint is a software package to perform Bayesian optimization according to specific algorithms (the “Software”). The Software is designed to automatically run experiments (thus the code name spearmint) in a manner that iteratively adjusts a number of parameters so as to minimize some objective in as few runs as possible.

**IMPORTANT:** Spearmint is under an **Academic and Non-Commercial Research Use License**.  Before using spearmint please be aware of the [license](LICENSE.md).  If you do not qualify to use spearmint you can ask to obtain a license as detailed in the [license](LICENSE.md).

**STEP 1: Installation**  

1. Install [python](https://www.python.org/), [numpy](http://www.numpy.org/), [scipy](http://www.numpy.org/). For academic users, the [anaconda](http://continuum.io/downloads) distribution is great. Use numpy 1.8 or higher.  
2. Download/clone the spearmint code  
3. Install the spearmint package using pip: `pip install -e \</path/to/spearmint/root\>` (the -e means changes will be reflected automatically)  
4. Download and install MongoDB: https://www.mongodb.org/   
5. Install the pymongo package using e.g., pip `pip install pymongo` or anaconda `conda install pymongo`  

**STEP 2: Setting up your experiment**  
1. Create a callable objective function. See ../examples/branin/branin.py as an example  
2. Create a config file. There are 3 example config files in the ../examples directory.  

**STEP 3: Running spearmint**  
1. Start up a MongoDB daemon instance:  
`mongod --fork --logpath <path/to/logfile\> --dbpath <path/to/dbfolder\>`  
2. Run spearmint: `python main.py \</path/to/experiment/directory\>`

**STEP 4: Looking at your results**  
Spearmint will output results to standard out / standard err. You can also load the results from the database and manipulate them directly. 
