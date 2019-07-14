#!/usr/bin/env bash

python3 --version

while IFS=";" read -r test_name model bash_size epoch lr activation layer nb_filtre final_activation
do
   echo "<----------------------------------------------------------------------------->\n"
   echo "<-------------------------------- $test_name --------------------------------->\n"
   echo "<----------------------------------------------------------------------------->\n"
   nvidia-smi
   horovodrun -np 4 -H localhost:4 python3 main.py $test_name $model $bash_size $epoch $activation $layer $nb_filtre $final_activation
   sleep 60
   echo "<----------------------------------------------------------------------------->"
   echo "<------------------------------------ FIN ------------------------------------>"
   echo "<----------------------------------------------------------------------------->"
done < ../Hyperparam/file_test.csv
