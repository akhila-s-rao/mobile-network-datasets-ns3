# mobile-network-datasets-ns3
This project contains scripts to generate datasets from 4G and 5G mobile networks. It comes packaged with the base ns-3 module along with additional third party ns-3 modules that are required to generate diverse datasets. 

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
 
## Step 6: Run the 5G hexagonal tesselation topology mobile network example 
./waf --cwd="logs" --run "lena-lte-comparison-user --scenario=UMi --numRings=1 --ueNumPergNb=30 --appGenerationTime=10000 --numerologyBwp=0 --simulator=5GLENA --trafficScenario=0 --randomSeed=1"





### Curated datasets genrated from these scripts can be found at (add google drive link) 
