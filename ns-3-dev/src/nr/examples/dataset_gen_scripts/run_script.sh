# First kill any simulations currently running
pkill ns3.35-cellular
sleep 1

# debug mode runs using gdb. Make sure to set the particular run number you want 
# (probably the one that crashed the quickest)
debug_mode=0

# If this is set to 2 then 2 separate simulation 
# campaigns will start one after the other 
num_sim_campaigns=1

# Save the directory location which has the run script and the simulations script
run_script_loc=$(pwd)

# Go to the home directory of ns3 where ./waf exists
cd ../../../../

script_save_dir_name="scripts_used_to_gen_this_data"

# ==============================
# This is the first set of runs 
# ==============================
# The location for saving data relative to ns-3-dev
#data_dir1="../../data_volume/lte_21macro_21Ue_delay_rtt_1Ue_dlThput" 
data_dir1="../../data_volume/logs"

# Saving all scripts and code used for this set of simulation runs
mkdir $data_dir1
echo "Saving all scripts and code used for this set of simulation runs in $data_dir1"
mkdir "$data_dir1/$script_save_dir_name"
# Save all relevant scripts (this script and the ns3 scripts)
cp -r "$run_script_loc/"* "$data_dir1/$script_save_dir_name/."

len=20
seed_shifter=0

if [ $debug_mode -eq 1 ]
then
  len=1
  #i=3
fi

for (( i=0; i<$len; i++ ))
do
  cmd_args="cellular-network-user \
	   --scenario=UMi \
	   --numRings=0 \
       --ueNumPerMacroGnb=5 \
       --useMicroLayer=true \
	   --numMicroCells=3 \
       --ueNumPerMicroGnb=10 \
	   --appGenerationTime=1000 \
	   --rat=LTE \
	   --operationMode=FDD \
       --handoverAlgo=A2A4Rsrq \
       --enableUlPc=true \
       --appDlThput=false \
       --appUlThput=false \
       --appHttp=false \
       --appDash=true \
       --appVr=false \
       --numMacroVrUes=1 \
       --numMicroVrUes=3 \
       --freqScenario=1 \
       --randomSeed=$(($i + $seed_shifter))"
       #--randomSeed=19"
       

  run_dir="run$(($i + 1))"  
  mkdir "$data_dir1/$run_dir"

  # Run in debug mode 
  if [ $debug_mode -eq 1 ]
  then
    # Run with gdb debug mode 
    ./waf --gdb --run-no-build "$cmd_args" --cwd="$data_dir1/$run_dir" 
    > "$data_dir1/$run_dir/simulation_info.txt"
    # Run with valgring debug mode
    #./waf --valgrind --run-no-build "$cmd_args" --cwd="$data_dir1/$run_dir"
    #./waf --run --command-template="valgrind --leak-check=full --show-reachable=yes %s" "$cmd_args --cwd=$data_dir1/$run_dir"
  else # Run
    taskset -c $i ./waf --run-no-build "$cmd_args" --cwd="$data_dir1/$run_dir" \
      > "$data_dir1/$run_dir/simulation_info.txt" \
      2> "$data_dir1/$run_dir/stderr_log.txt" &
  fi

  echo "Started run $(($i + 1))" 
  sleep 2
done



# ==============================
# This is the second set of runs 
# ==============================

if [ $num_sim_campaigns -eq 2 ]
then

echo "SECOND SET OF RUNS"

# The location for saving data relative to ns-3-dev
#data_dir2="../../data_volume/lte_21macro_21Ue_delay_rtt_1Ue_ulThput" 
data_dir2="../../data_volume/logs2"
mkdir $data_dir2
# Saving all scripts and code used for this set of simulation runs
echo "Saving all scripts and code used for this set of simulation runs in $data_dir2"
mkdir "$data_dir2/$script_save_dir_name"
# Save all relevant scripts (this script and the ns3 scripts)
cp -r "$run_script_loc/"* "$data_dir2/$script_save_dir_name/."

len=10
seed_shifter=150
#i=3
for (( i=0; i<$len; i++ ))
do
  cmd_args="cellular-network-user \
	   --scenario=UMi \
	   --numRings=0 \
       --ueNumPerMacroGnb=5 \
       --useMicroLayer=true \
	   --numMicroCells=3 \
       --ueNumPerMicroGnb=10 \
	   --appGenerationTime=100 \
	   --rat=LTE \
	   --operationMode=FDD \
       --handoverAlgo=A2A4Rsrq \
       --enableUlPc=true \
       --appDlThput=false \
       --appUlThput=false \
       --appHttp=true \
       --appDash=true \
       --appVr=true \
       --numMacroVrUes=1 \
       --numMicroVrUes=3 \
       --freqScenario=1 \
       --randomSeed=$(($i + $seed_shifter))"

  run_dir="run$(($i + 1))"  
  mkdir "$data_dir2/$run_dir"

  # Run 
  taskset -c $(($i + 10)) ./waf --run-no-build "$cmd_args" --cwd="$data_dir2/$run_dir" \
    > "$data_dir2/$run_dir/simulation_info.txt" & #\
    #2> "$data_dir2/$run_dir/stderr_log.txt" &

  echo "Started run $(($i + 1))" 
  sleep 2
done




fi

