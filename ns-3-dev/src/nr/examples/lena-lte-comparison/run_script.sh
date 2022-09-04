# First kill any simulations ciurrently running
pkill ns3.35-lena-lte
sleep 1
# If this is set to 2 then 2 separate simulation 
# campaigns will start one after the other 

num_sim_campaigns=1
run_script_loc=$(pwd)
# Go to the home directory of ns3 where ./waf exists
cd ../../../../
# ==============================
# This is the first set of runs 
# ==============================
len=20
#i=2
for (( i=0; i<$len; i++ ))
do
  cmd_args="lena-lte-comparison-user \
	   --scenario=UMi \
	   --numRings=0 \
	   --numMicroCells=3 \
	   --ueNumPergNb=5 \
	   --useMicroLayer=true \
	   --appGenerationTime=100 \
	   --simulator=LENA \
	   --operationMode=FDD \
           --randomSeed=$i"

  run_dir="run$(($i + 1))" 
  data_dir="../../data_volume/logs"  
  mkdir "$data_dir/$run_dir"

  # Without gdb debug mode 
  taskset -c $i ./waf --run-no-build "$cmd_args" --cwd="$data_dir/$run_dir" \
    > "$data_dir/$run_dir/console_log.txt" &

  # With gdb debug mode 
  #./waf --gdb --run-no-build "$cmd_args" --cwd="$data_dir/$run_dir" 
  #> "$data_dir/$run_dir/console_log.txt"

  # With valgring debug mdoe
  #./waf --valgrind --run-no-build "$cmd_args" --cwd="$data_dir/$run_dir"
  #./waf --run --command-template="valgrind --leak-check=full --show-reachable=yes %s" "$cmd_args --cwd=$data_dir/$run_dir"


  echo "Started run $(($i + 1))" 
  echo "Saving input parameters used for run by saving this run_script.sh"
  cp $run_script_loc/run_script.sh "$data_dir/$run_dir/."
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

