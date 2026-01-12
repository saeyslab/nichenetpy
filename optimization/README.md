# Optimization of the source weights
To run on the ugent high performance computer
```
# select cluster
module swap cluster/doduo
# set up virtual environment
qsub nichenetpy/install_venv.pbs -v cluster=doduo
# run the optimization (you can customize the optimization procedure by altering the script)
qsub nichenetpy/run_optimization.pbs -v cluster=doduo
```

# Construct model after optimization
The optimization will have one log per fold as a result. `construct_final_model_from_logs.py` processes these log files into a nichenet model. A shell script is provided as an example of how to use this script, see `construct_model_from_logs/construct_v2.sh`. You can build a mouse model using the same source weights if you convert the networks using `../scripts/networks_human2mouse.py`. Note that the substring "human" will be replaced by "mouse" in the filename. 
Illustrative example:
```
python C:/Users/victorm/Documents/nichenetpy/scripts/networks_human2mouse.py \
	D:/Data/nichenetpy/v3/networks/mouse \
	--network D:/Data/nichenetpy/v3/networks/human/lr_network_human.csv \
    --network D:/Data/nichenetpy/v3/networks/human/gr_network_human.csv \
    --network D:/Data/nichenetpy/v3/networks/human/sig_network_human.csv
```
