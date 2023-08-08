len=20
for (( i=1; i<=$len; i++ ))
do
    echo run$i
    tail -n 6 ../../data_volume/logs_today/run$i/simulation_info.txt
    #tail -n 6 ../../data_volume/logs/run$i/stderr_log.txt
done

