# PGC-NN

## Description

This repository presents PGC-NN: a graph neural network able to semantically enriching Points of Interest (POIs) according to spatiotemporal data.

This approach is a piece of a more complete work [1], removed from the original repository [2].

## Data

The required data is in folder processed_data, maded by processing_data.ipynb notebook. This notebook do pre-processing to the original data (more info in [1]), limiting the number of users to 100. This decision was taked to favor perfomance.

Whereas Deep Learning performance better with large dataframes, in real tests, may be necessary execute with more data.

<span style="color: red;">**OBS: The Data was removed to not overload the repository, but [1] contain a detailed explanation about them.**</span>

## Requirements and Execution

### Requirements

* Python 3
    * [link](https://www.python.org/downloads/)
* virtualenv
    * `$ pip install virtualenv`

### Execution

To execute this project, first you need to create and activate your virtual environment. The step-by-step is:

<span style="color: red;">**OBS: The follow step-by-step was created to use in Linux Operation System.**</span>

1. Create your virtual environment
    * `$ virtualenv myenv`
    * Replace 'myenv' with a name of your choice

2. Activate the virtual environment
    * `$ source myenv/bin/activate`

3. Update pip
    * `$ pip install --upgrade pip`

4. Install requirements
    * `$ pip install -r requirements.txt`

5. Execute
    * `$ ./run.sh categorization`

***

## References

[1] Cláudio G.S. Capanema, et al. Combining recurrent and Graph Neural Networks to predict the next place’s category. Ad Hoc Netw. 138, C (Jan 2023). https://doi.org/10.1016/j.adhoc.2022.103016

[2] CLAUDIOCAPANEMA. poi_gnn. Disponível em: <https://github.com/claudiocapanema/poi_gnn>.

***