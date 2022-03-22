if [ $# -ne 2 ]; then
       echo "Please enter start run number and start core index"
       exit 0
fi

# Please change this as required on your system
log_loc="/home/akhilarao/data_from_ns3_dash_simulation"
# Range of max delay values for the start time of video requests on UEs
#maxdelay=(1 5 10 15 20 25 30 35 40 45 50) # seconds
maxdelay=(1 5 10) # seconds


echo "log location is $log_loc"
sleep 2
len=${#maxdelay[@]}
echo "num of runs = "$len" start run is "$1
sleep 2
# Takes you from the src/dash/examples folder to the main ns3 folder where waf is. 
cd ../../../

# Since sometime we change this parameter, set it before each run.
framesperseg=100
sed -i '36s/.*/#define MPEG_FRAMES_PER_SEGMENT '${framesperseg}'/' src/dash/model/mpeg-header.h

# save this script so we know the parameter settings made
cp src/dash/examples/vary_startTime.sh ${log_loc}'/.'


for (( i=0; i<$len; i++ ))
do
   echo ${maxdelay[i]}
   mkdir ${log_loc}/run$(($i + $1))
   if [ $i -eq 0 ]; then
      run_type='--run'
      sleep_time=10 
   else
      run_type='--run-no-build'
      sleep_time=3
   fi

   cmd_args="src/dash/examples/lena-dash-ran-metrics \
--simTime=6 \
--randSeed=13 \
--epc=true --useUdp=false --algorithms='ns3::AaashClient' \
--bufferSpace=10000000 \
--homeEnbDeploymentRatio=0.2 --macroUeDensity=0.000075 \
--outdoorUeMinSpeed=1.4 --outdoorUeMaxSpeed=14.0 \
--maxStartTimeDelay=${maxdelay[i]} \
"
   taskset -c $(($i + $2)) ./waf "$run_type" "$cmd_args" --cwd=${log_loc}'/run'$(($i + $1)) \
> ${log_loc}'/run'$(($i + $1))/dash_client_log.txt 2>&1 &
   
   sleep $sleep_time
done

