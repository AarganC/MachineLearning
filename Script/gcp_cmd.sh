#Auth gcloud
gcloud auth list

# Connection ssh if the tools gcp terminal not work and if I not arrived to connect in ssh classic
gcloud compute ssh --project theta-anchor-246215 --zone europe-west4-a instance-2
#gcloud compute ssh --project [PROJECT_ID] --zone [ZONE] [INSTANCE_NAME]


# Copy file
#gcloud compute scp [LOCAL_FILE_PATH] [INSTANCE_NAME]:~
gcloud compute scp [LOCAL_FILE_PATH] [INSTANCE_NAME]:~


# Connect to jupyter notebook
source activate tensorflow
jupyter-notebook --no-browser --port=5000  --ip=10.164.0.2

# Stop INSTANCE dans l'instances
gcloud compute instances stop instance-2 -q --zone europe-west4-a

# Run jupyter
export PATH="/home/aargancointepas/Anaconda/anaconda3/bin:$PATH"
source .bashrc
source activate tensorflow
jupyter-notebook --no-browser --port=5000  --ip=10.164.0.2
http://34.90.87.249:5000/

# Terminal
TMUX
I3
