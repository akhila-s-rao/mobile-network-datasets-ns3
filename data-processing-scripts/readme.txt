#=======================================
# Sanity checks that this script runs 
#=======================================

# (done) Runtime: Did all logs for all runs generate data for the entire simulation duration (prints)

# (done) IDs: ue_id / cell_id presence in logs. Does each log have representation from
#      all the UEs and cells that should be present (prints)

# (done) Delay histograms: Are there any ridiculously large delays in the data that cannot be explained. 
#                   (delay, rtt trace histograms combined for all UEs)

# (done) Thput histograms: Understand the throuphput variability caused by only signal variations and not user contention  
#                   ul/dl histograms combined for all UEs  

# (done) Byte Matrix: Are the number of bits going over the network at the different layers similar ? : (byte matrix)
#              Is the traffic UL and DL as expected based on the traffic being sent. 
#              Is it symetric when there is no one way traffric ? : (byte matrix)

# (done) Mobility plot: Plot the movement of the UEs and make sure they make sense (In another script)

# (done) Delay probe delivery rate: Combined for all UEs to check how droppy this network is
# (done) Average throughput over the entire region (all UEs, all cells, entire area)  
# (done) Number of handovers per unit time per unit cell
# (done) Average/histogram time between HO for a slow moving UE and a fast moving UE

# (done) plot per cell timeseries of UL/DL traffic aggrergated over windows (could compare with the real trace traffic from Fehmi)
# (done) plot per UE time series of DL SINR/RSRP separately for a sample fast and slow moving UE   

# (done) Plot distance to BS versus delay/rtt: combined for all UEs
# (done) Plot distance to BS versus ul/dl throughput: combined for all UEs
# (done) Plot distance to BS versus signal strength: combined for all UEs (rsrp, ul sinr, dl sinr) NOTE: power control is being used 

# Application metrics aggregated over all UEs running this app to plot histograms 
# (done) page_load_time histogram: page load time over all webpages viewed 
# (done) segment_bitrate histogram: bitrate of segment requested over all videos watched
# (done) vr_burst_time histogram: time to receive a full burst in VR



Scenario 1
==========
Macro topology 
5 UEs per enb no VR  
slow and fast UEs

Scenario 2
==========
Macro + micro topology (30 dbm for macro and 20 dBm for micro, use 10dBm for UE with auto)
30 Ues per enb with VR 
Slow and fast UEs
create a bounding box around the micro enbs and make some UEs move very slow move only within it
      - Other UEs can pass this box and connect to the micro UEs of course but there are some UEs that can be separated by IMSI that only connect with the micro UE 
      - Micro box size would be 100 m by 100m. Don't force the UEs in the box to connecyt tp micrp. It should be decide dby handover   


Scenario 3
==========
macro topology 
Rural ISD 
rural channel model 
slow and fast UEs 


compare scenario 1 and scenario 2 
separate plots for slow and fast UEs in scenario2 
separate plots for macro connected UEs and micro connected UEs in scenario 2




Things I need to find out 
=========================
- How does the scheduler handle difference bearer types, does it do QoS based scheduling ? If not then what is the incentive for declaring bearers and not using the default bearer for all traffic streams ? 
- 







==================================================================================
ul tx power: max 10dBm with power control 

I should not remove x2 connection from micro UEs
I should not force initial connections 


-----------------------
macro tx power: 15 dBm 
macro #UEs: 15

ul rx rate: 0.97 
dl rax rate: 0.93 
------------------------
macro tx power: 23 dBm 
macro #UEs: 15

ul rx rate: 0.97 
dl rax rate: 0.99
------------------------
The UL rx rate in interference llimited and hence will only reduce beyong this if #UEs are reduced
So In the final scenario with #UEs beign increased the ul rs rate will actually increase

The Dl rx rate at 23 dBm is pretty high already, we can increase the macro tx power to 25 dBm to make it 1.00 
This however I expect will not do anything to the UL rx rate

macro tx power: 
macro #UEs:
micro tx power:
micro #UEs:
