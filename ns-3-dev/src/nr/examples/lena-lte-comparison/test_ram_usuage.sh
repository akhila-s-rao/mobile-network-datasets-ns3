len=10000
for (( i=0; i<$len; i++ ))
do
  free -m >> ram_usage.log
  free -m
  sleep 30
done
