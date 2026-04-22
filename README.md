# avoidance
clone the avoidance repository and all its submodules:
```bash
git clone --recursive https://github.com/Fanxing-LI/avoidance
```

# install VisFly
```bash
cd avoidance/VisFly
```
Follow the instructions in [VisFly](https://github.com/SJTU-ViSYS-team/VisFly) to install the VisFly simulator.

# clone the datasets
```bash
cd avoidance/VisFly/datasets
git clone -b avoidance https://huggingface.co/datasets/LiFanxing/visfly-beta
```

# train the policy
```bash
cd avoidance
conda activate visfly
python exps/real_world/run.py -t 1 -a SHAC -e navigation 
# python exps/real_world/run.py -t 1 -a PPO -e navigation -c comment
python exps/real_world/run.py -t 0 -a SHAC -e navigation -w SHAC_std_1.zip
```
If you define a new comment, replace your comment with `std` in the last command.

# download the pre-trained weights
[Download](https://drive.google.com/file/d/1vKx2L6aIMCFVegsi_5F9uyyCsQg7XDZm/view?usp=sharing) the pre-trained weights and put it in `avoidance/exps/real_world/saved/navigation/`.
```bash
cd avoidance
conda activate visfly
python exps/real_world/run.py -t 0 -a SHAC -w checkpoint.zip -e navigation

    
