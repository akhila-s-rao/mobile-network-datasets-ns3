len=70
for (( i=1; i<=$len; i++ ))
do
    echo run$i
    tail -n 6 ../../data_volume/30_11_power_1000_5_16/run$i/simulation_info.txt
done

