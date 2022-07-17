# If this is set to 2 then 2 separate simulation 
# campaigns will start one after the other 

num_sim_campaigns=1
run_script_loc=$(pwd)
# Go to the home directory of ns3 where ./waf exists
cd ../../../../
# ==============================
# This is the first set of runs 
# ==============================

len=10
for (( i=0; i<$len; i++ ))
do
  cmd_args="lena-lte-comparison-user \
	   --scenario=UMi \
	   --numRings=0 \
	   --ueNumPergNb=2 \
	   --appGenerationTime=1000 \
	   --numerologyBwp=0 \
	   --simulator=LENA \
	   --operationMode=FDD \
	   --trafficScenario=0 \
           --randomSeed=$i"

  run_dir="run$(($i + 1))"  
  mkdir ../../logs/$run_dir

taskset -c $i ./waf --run-no-build "$cmd_args" --cwd="../../logs/$run_dir" \
  > "../../logs/$run_dir/console_log.txt" &

#./waf --run-no-build "$cmd_args" --cwd="../../logs/$run_dir" --command-template="gdb %s"




  echo "Started run $(($i + 1))" 
  echo "Saving input parameters used for run by saving this run_script.sh"
  cp $run_script_loc/run_script.sh "../../logs/$run_dir/."
  sleep 2
done



# ==============================
# This is the second set of runs 
# ==============================

if [ $num_sim_campaigns -eq 2 ]
then

len=10
for (( i=0; i<$len; i++ ))
do
  cmd_args="lena-lte-comparison-user \
           --scenario=UMi \
           --numRings=1 \
           --ueNumPergNb=30 \
           --appGenerationTime=1000 \
           --numerologyBwp=0 \
           --simulator=5GLENA \
           --trafficScenario=0 \
           --randomSeed=$i"

  run_dir="run$(($i + 1))"  
  mkdir ../../logs/$run_dir

taskset -c $i ./waf --run-no-build "$cmd_args" --cwd="../../logs/$run_dir" \
  > "../../logs/$run_dir/console_log.txt" &

  echo "Started run $(($i + 1))" 
  echo "Saving input parameters used for run by saving this run_script.sh"
  cp $run_script_loc/run_script.sh "../../logs/$run_dir/."
  sleep 2
done




fi

