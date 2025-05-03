# Final Project for CS485: Deep Learning on Graphs

## Instructions for Reproducing Results

Make sure you have Python 3.12.9. Start off with running the following to download dependencies:
```
pip install -r requirements.txt
```
To collect data manually, first, adjust the `tokens` array at the top of the `data_collection.py` script. Then adjust the arguments and paths at the top of script as needed (or leave as defaults) and then run the script.
```
python data_collection.py
```
If anything fails during the web scraping or RPC data collection process just rerun the script as it keeps logs of everything needing to be collected.

You can also skip data collection as heterographs are already provided in this repository. If you want to build the graphs after collecting data manually using the script then use the `heterograph.py` script. 
```
python heterograph.py
```
Otherwise, you can directly train and test our models by running the `hgat.ipynb`, `hetero_gnn.ipynb`, and `r_gcn.ipynb` notebooks. 

The other files and folders not mentioned above in this repository are tests, logs, and temporary files. Feel free to explore them if you are curious.
