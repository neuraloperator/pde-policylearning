# pde-observers
1. Uncomment gt and set collect_data=true to generate data.
```
python run_pde_observers.py --train_yaml configs/base_rno.yaml
```

2. Set data_folder to 'outputs/{exp_name}' to train rno and evaluate. The command is the same.
```
python run_pde_observers.py --train_yaml configs/base_rno.yaml
```

# old instructions for reference.
1. First run the following to preprocess data (for efficient running and debugging):

```
python lib/mat2npy.py
```


2. Then run to train the supervised pde observers:

```
python run_pde_observers.py
# rno
python run_pde_observers.py --train_yaml configs/base_rno.yaml
python run_pde_observers.py --train_yaml configs/base_transformer.yaml
```


3. Run the controller via:

```
python run_control.py
```