# Local Binary Patterns: Overview and Application on Helicobacter Pylory

Contains all the useful scripts for performing experiments with LBP and its variants.

Note: Run the following commands in the project's root directory

```bash
# Install Virtual Enviroment (Optional but high recommended):

  $ pip install virtualenv (or `pip3 install virtualenv`)

# Create a vitural enviroment for experimentation:

  $ virtualenv venv_lbp

# Activate the enviroment created

  $ source venv_lbp/bin/activate

# Install Librarys requireds:

  $ pip install -r requirements.txt

# Build the LBP library:

  $ python setup.py build_ext --inplace
```

Example of use

```bash
# If you need of some help, run:
  $ python exec_run.py -h
  
  usage: exec_run.py [-h] --dataset DATASET --variant VARIANT [--method METHOD]
                     [--points POINTS] [--radius RADIUS]
                     [--size_train SIZE_TRAIN] [--output OUTPUT] [--load LOAD]

  optional arguments:
    -h, --help            show this help message and exit
    --dataset DATASET, -d DATASET
                          Dataset that contains the images
    --variant VARIANT, -v VARIANT
                          Descritor LBP variant accepted: base_lbp,
                          improved_lbp, extended_lbp, completed_lbp, hamming_lbp
    --method METHOD, -m METHOD
                          Method Accepted: `nri_uniform` and `uniform`
    --points POINTS, -p POINTS
                          Number of points at neighboorhood
    --radius RADIUS, -r RADIUS
                          Radius of points at neighboorhood
    --size_train SIZE_TRAIN, -s SIZE_TRAIN
                          Length of train dataset
    --output OUTPUT, -o OUTPUT
                          Path to output results
    --load LOAD, -l LOAD  Save descriptors computed

# For example, when running:
  $ python exec_run.py -d path_dataset -v completed_lbp -m uniform -p 8 -r 1 -s 0.8 -o results -l False
  
The code will perform the calculation of the Completed LBP descriptor on the ´path_dataset´ dataset using the uniform and invariant method of rotation, parameters P and R equal to 8 and 1, respectively, separating a total of 80% for training and the rest for testing .

The ´-o´ parameter indicates a folder where the results will be stored. If this folder does not exist, the program will automatically create it.

The parameter -l is a boolean indicating whether you want to load the calculated descriptors in a .txt file so that it is not necessary to recalculate them. If you are going to run the experiment for the first time, this argument should be ignored.

```
