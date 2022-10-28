len=20
for (( i=1; i<=$len; i++ ))
do
    echo run$i
    tail -n 6 logs/run$i/console_log.txt
done

