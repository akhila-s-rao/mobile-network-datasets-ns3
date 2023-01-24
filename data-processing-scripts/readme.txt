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



#=======================================
# Organising UEs into groups 
#=======================================

# Slow moving UEs group 
# Fast moving UEs group 
# Web browsing UEs group 
# Video streaming UEs group 
# VR UEs group 
# No traffic apps UEs group 
# Thput measurement UE 

# based on the UE entries in the corresponding log files I can get the UE IMSIs of app groups  
# it is only slow moving and fast moving UE categories that I cannot separate out from files 
# (easily at least, I could measure the average speed and categorize them)


# Separate out the IDs of the UEs that have a specific set of apps running on them  
#ueIds = np.arange(0,total_num_ues)
# UEs with only delay measurements and no background traffic:
#isIncluded=np.array( [((x % 5) == 0) for x in range(0,total_num_ues)], dtype=bool)
#trafficClass1_ueIds=ueIds[isIncluded]
# UEs with video streams:
#isIncluded=np.array( [ ( ((x % 5) != 0) and (x%2 ==0) ) for x in range(0,total_num_ues)], dtype=bool)
#trafficClass2_ueIds=ueIds[isIncluded]
# UEs with web browsing:
#isIncluded=np.array( [ ( ((x % 5) != 0) and (x%2 !=0) ) for x in range(0,total_num_ues)], dtype=bool)
#trafficClass3_ueIds=ueIds[isIncluded]

#!!!!!! But I do not know what the IMSI of these ueIds are 
#Maybe I should create these groups in ns3 code using IMSI instead of ueId 




Plan
====
Tuesday
- Decide on parameters for the 2 scenarios and create a 200s run for each 
- Take those graphs and put it in the paper for sizing
- Make the plots pretty (take away space between them, adjust the axis etc.)
- Organize and pretty up the scenario description tables

Wednesday 
- Reduce size of motivation and related work 
- Complete the data format section
- Write the design decisions section
- Add the ML tasks in related wrok table
- Read the beyond throughput paper and see what inspirations I can borrow. 

Thursday 
- Pimp up the github page and add links to the google directory with data 
- Write about how this is useful to the multimedia community

Friday 
- Improve language 








parameters yet to be fixed 
==========================

- DONE ueTxPower (lte-utils.cc) enableUlPc but the default value set to 10 dBm 
-  ? macro txPower (lte-utils.cc) 30 dBm (we set it to lower than the dual-stripe scenario even though I think we have a more punishing model, is this okay ? I need to check with a REM map)
- ? microTx power (cellular-network.h) right now it is 18dBm 
- ? cosine antenna MaxGain (lte-utils.cc) as of now it is 3  
- DONE handover algorithm (cellular-network.cc) A2A4Rsrq
- DONE beamwidth for cosine antenna (lte-utils.cc) 60 degrees 
- ? ISD 500 m depends on the Tx power set for macro Tx
- Macro bounding box 
- Micro bounding box
- 





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




