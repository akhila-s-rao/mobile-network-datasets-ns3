len=10
for (( i=1; i<=$len; i++ ))
do
    echo run$i
    tail -n 6 ../../data_volume/lte_3macro_15Ue_delay_rtt_dash/run$i/simulation_info.txt
done

