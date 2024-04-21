#!/bin/sh 
 
# Install dependencies
pip install flexs==0.2.8
pip install adapt-diagnostics==1.3.0 
pip install tensorflow==2.11.0
pip install fastaparser==1.1
pip install matplotlib==3.3.4

# Pip install the package
python -m pip install ./
