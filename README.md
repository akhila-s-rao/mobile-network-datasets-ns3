# mobile-network-datasets-ns3
This project contains scripts to generate datasets from ns-3 4G and 5G mobile networks. It comes packaged with the base ns-3 module along with additional third party ns-3 modules that are required to generate diverse datasets. 

It also contains scripts to parse the data, do sanity checks on them and visualize it in various ways.  

# Install instructions
This is a full ns-3 installation along with the additional modules and changes to the base code. 

## Step 1: Clone the repository 
git clone https://github.com/akhila-s-rao/mobile-network-datasets-ns3.git

## Step 2: Go into the ns-3 code folder 
cd mobile-network-datasets-ns3/ns-3-dev 

## Step 3: Configure the ns-3 code
./waf configure --build-profile=optimized --enable-examples 

This creates an optimized build which disables debug mode and improves simulation speed.
To enable debug mode which allows you to print logs from the NS_LOG_COMPONENT 

./waf configure --enable-examples

Several components will not be configured and be shown in red. This is normal. 

## Step 4: Build the code
./waf build

## Step 5: Create a folder to store your logs
mkdir logs
 
## Step 6: Tune parameter knobs, modify the simulation scenario setup and run a simulation campaign   
cd mobile-network-datasets-ns3/ns-3-dev/src/nr/examples/dataset_gen_scripts 
edit run_script.sh as required
bash run_script.sh 

### Curated datasets generated from these scripts can be found at (google drive link)
https://drive.google.com/drive/folders/1-OQolvhK1mpFNFVLF4XTC_c3_EjLPwz-?usp=sharing


## Acknowledgement
This work was funded by the H2020 AI@EDGE project (grant. no. 101015922)

