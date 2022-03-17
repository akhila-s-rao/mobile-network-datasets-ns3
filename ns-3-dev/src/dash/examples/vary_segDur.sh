if [ $# -ne 2 ]; then
       echo "Please enter start run number and start core index"
       exit 0
fi

# Please change this as required on your system
log_loc="/home/akhilarao/data_from_ns3_dash_simulation/vary_segDur"
# Range of segment duration 
framesperseg=(100 200 400 800 1600)

echo "log location is $log_loc"
sleep 2
len=${#framesperseg[@]}
echo "num of runs = "$len" start run is "$1
sleep 2
# Takes you from the src/dash/examples folder to the main ns3 folder where waf is. 
cd ../../../

# savbe this script so we know the parameter settings made
cp src/dash/examples/vary_segDur.sh ${log_loc}'/.'

for (( i=0; i<$len; i++ ))
do
   sed -i '36s/.*/#define MPEG_FRAMES_PER_SEGMENT '${framesperseg[i]}'/' src/dash/model/mpeg-header.h

   echo ${framesperseg[i]}
   mkdir ${log_loc}/run$(($i + $1))
   run_type='--run'
   sleep_time=5

cmd_args="src/dash/examples/lena-dash-ran-metrics \
--simTime=6 \
--randSeed=13 \
--epc=true --useUdp=false --algorithms='ns3::AaashClient' \
--bufferSpace=10000000 \
--homeEnbDeploymentRatio=0.2 --macroUeDensity=0.000075 \
--outdoorUeMinSpeed=1.4 --outdoorUeMaxSpeed=14.0 \
--maxStartTimeDelay=5 \
"

taskset -c $2 ./waf "$run_type" "$cmd_args" --cwd=${log_loc}'/run'$(($i + $1)) \
> ${log_loc}'/run'$(($i + $1))/dash_client_log.txt 2>&1

   sleep $sleep_time
done
