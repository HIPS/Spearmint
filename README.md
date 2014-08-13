spearmint: Bayesian optimization codebase
=========================================

**STEP 1: Installation**  

1. Install python, numpy, scipy. For academic users, the anaconda distribution is great. Use numpy 1.8 or higher. 
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
