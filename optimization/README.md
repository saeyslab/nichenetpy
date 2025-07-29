# Optimization of the source weights
To run on the ugent high performance computer
```
# select cluster
module swap cluster/doduo
# set up virtual environment
qsub nichenetpy/install_venv.pbs -v cluster=doduo
# run the optimization (you can customize the optimization procedure by alterring the script)
qsub nichenetpy/run_optimization.pbs -v cluster=doduo
```
