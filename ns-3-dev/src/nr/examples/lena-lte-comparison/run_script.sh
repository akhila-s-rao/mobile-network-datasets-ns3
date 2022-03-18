# We need to go to the home directory of ns3  
cd ../../../../
pwd

len=5
#for i in 7 8 13 14 15 
for (( i=0; i<$len; i++ ))
do
  cmd_args="lena-lte-comparison-user \
	   --scenario=UMi \
	   --numRings=1 \
	   --ueNumPergNb=10 \
	   --appGenerationTime=10000 \
	   --numerologyBwp=0 \
	   --simulator=LENA \
	   --trafficScenario=2 \
           --randomSeed=$i"

  run_dir="run$(($i + 1))"  
  mkdir logs/$run_dir

  ./waf --run-no-build "$cmd_args" --cwd="logs/$run_dir" \
  > "logs/$run_dir/console_log.txt" &

  echo "Started run $(($i + 1))" 

  #cp run_script.sh "logs/$run_dir/."
  sleep 2
done
