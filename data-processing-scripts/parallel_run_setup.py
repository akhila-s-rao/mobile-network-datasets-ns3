from multiprocessing import Pool
import subprocess
# This script was created to use instead of run_script.sh to monitor the outcome of each run and start a fresh run if any runs crash or completes. This way I can always keep the N cores I have occupied by setting a large num of runs. I can always kill the script to prevent new ones from launching if I am satisfied with how many runs I have 

# start one single run
def setup_run(rand_index, data_dir):
    # Go over all to make sure it is being setup as needed 
    cmd_args=("cellular-network-user "
    +"--scenario=UMi "
    +"--numRings=0 "
    +"--ueNumPerMacroGnb=10 "
    +"--useMicroLayer=true "
    +"--numMicroCells=3 "
    +"--ueNumPerMicroGnb=20 "
    +"--appGenerationTime=1000 "
    +"--rat=LTE "
    +"--operationMode=FDD "
    +"--handoverAlgo=A2A4Rsrq "
    +"--enableUlPc=true "
    +"--appDlThput=false "
    +"--appUlThput=false "
    +"--appHttp=true "
    +"--appDash=true "
    +"--appVr=true "
    +"--numMacroVrUes=1 "
    +"--numMicroVrUes=3 "
    +"--freqScenario=1 "
    +"--randomSeed="+str(rand_index))

    run_dir="run"+str(rand_index+1)
    result = subprocess.run(['mkdir', data_dir+'/'+run_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stderr)
    print(result.stdout)
    print('Starting run ', rand_index+1)
    result = subprocess.run(['cd ../ns-3-dev; ./waf --run-no-build '+'"'+cmd_args+'"'+
                             ' --cwd='+data_dir+'/'+run_dir+' > '+data_dir+'/'+run_dir+'/'+'simulation_info.txt '+
                             ' 2> '+data_dir+'/'+run_dir+'/'+'simulation_err.txt'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print('Finished run ', rand_index+1)
    print(result.stderr)
    print(result.stdout)
    #sleep 2

# total number of runs to create 
runs=20
run_script_loc="../ns-3-dev/src/nr/examples/dataset_gen_scripts"
script_save_dir_name="scripts_used_to_gen_this_data"
data_dir="../../data_volume/logs_today"

# Create the data directory
result = subprocess.run(['mkdir', data_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(result.stderr)
print(result.stdout)

# Echo that I have done so  
result = subprocess.run(['echo', "Saving all scripts and code used for this set of simulation runs in", data_dir], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(result.stderr)
print(result.stdout)

# Create the script save directory inside the data directory 
result = subprocess.run(['mkdir', data_dir+'/'+script_save_dir_name], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(result.stderr)
print(result.stdout)

# Save all the C++ ns3 scripts that create and run the scenario into this directory
print(run_script_loc+'/*')
print(data_dir+'/'+script_save_dir_name+'/.')
result = subprocess.run(['cp -r '+run_script_loc+'/* '+data_dir+'/'+script_save_dir_name+'/.'], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
print(result.stderr)
print(result.stdout)

# Save this script as well so that I know what the input parameter settings for these runs are
result = subprocess.run(['cp -r '+'parallel_run_setup.py '+data_dir+'/'+script_save_dir_name+'/.'], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
print(result.stderr)
print(result.stdout)

pool = Pool(processes=20)
inputs = [(x, data_dir) for x in range(0,runs)]
outputs = pool.starmap(setup_run, inputs)    
