if [ $# -ne 2 ]; then
       echo "Please enter start run number and start core index"
       exit 0
fi

# Please change this as required on your system
log_loc="/home/akhilarao/data_from_ns3_dash_simulation/vary_num_macroues"
# Range of macroue_density
macroue_density=(0.000005 0.000015)
#macroue_density=(0.000005 0.000015 0.000025 0.000035 0.000045 0.000055 0.000065 0.000075 0.000085 0.000095)

echo "log location is $log_loc"
sleep 2
len=${#macroue_density[@]}
echo "num of runs = "$len" start run is "$1
sleep 2
# Takes you from the src/dash/examples folder to the main ns3 folder where waf is. 
cd ../../../

# Since sometime we change this parameter, set it before each run.
framesperseg=100
sed -i '36s/.*/#define MPEG_FRAMES_PER_SEGMENT '${framesperseg}'/' src/dash/model/mpeg-header.h

# save this script so we know the parameter settings made
cp src/dash/examples/vary_macroUeDensity.sh ${log_loc}'/.'

for (( i=0; i<$len; i++ ))
do
   echo ${macroue_density[i]}
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
--homeEnbDeploymentRatio=0.2 --macroUeDensity=${macroue_density[i]} \
--outdoorUeMinSpeed=1.4 --outdoorUeMaxSpeed=14.0 \
--maxStartTimeDelay=5 \
"

taskset -c $(($i + $2)) ./waf "$run_type" "$cmd_args" --cwd=${log_loc}'/run'$(($i + $1)) \
> ${log_loc}'/run'$(($i + $1))/dash_client_log.txt 2>&1 &

   sleep $sleep_time
done
