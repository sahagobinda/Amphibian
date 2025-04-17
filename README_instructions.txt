# Install the required packages:

pip install -r requirements.txt

# Dataset for CIFAR-100 and 5-Datasets experiments will be automatically downloaded in "data" folder

# For TinyImageNet, first give the correct permissions to the download script:

chmod +x download_tinyimgnet.sh

# Then, run the following command to download and process the data for use by Pytorch Dataloader:

./download_tinyimgnet.sh

# After successful installations and data processing run: 

source run_amphibian_experiments.sh

# This will run Split CIFAR-100, Split TinyImagenet and Split 5-Datasets experiments reproted in the paper for our Amphibian method. 
# At the end of each experiment numbers for ACC and BWT metrics will be shown in the terminal. 




