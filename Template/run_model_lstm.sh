#!/usr/bin/env bash

python3 --version

python3 Notebook-Cointepas_Aargan.py
sleep 60
gcloud compute instances stop instance-2 -q --zone europe-west4-a
