#!/bin/bash
# Install pip packages
pip install --upgrade pip
pip install --only-binary=:all: numpy pandas scikit-learn scipy
pip install -r requirements.txt
