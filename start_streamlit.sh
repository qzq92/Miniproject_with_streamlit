#!/usr/bin/bash

echo "Activate miniproject env"
eval "$(conda shell.bash hook)"
conda activate miniproject

streamlit run Landing_Page.py