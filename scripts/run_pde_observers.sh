python run_pde_observers.py --train_yaml configs/matlab_rno.yaml

# process data
python lib/mat2npy.py

python run_cfd_simulation.py


