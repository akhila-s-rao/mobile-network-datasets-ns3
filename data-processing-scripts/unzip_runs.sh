for entry in lte_3Cell_6UEs_delayTrace_rttTrace_smallHttpTrace_withHandover_scenario1_2/*.zip
do
  echo "unzipping $entry"
  unzip $entry &
done
