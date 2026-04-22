# Avoidance
clone the avoidance repository and all its submodules:
```bash
git clone --recursive https://github.com/Fanxing-LI/avoidance
```

# Install VisFly
```bash
cd avoidance/VisFly
```
Follow the instructions in [VisFly](https://github.com/SJTU-ViSYS-team/VisFly) to install the VisFly simulator.

# Clone the datasets
```bash
cd avoidance/VisFly/datasets
git clone -b avoidance https://huggingface.co/datasets/LiFanxing/visfly-beta
```

# Train the policy (Or skip this step and download pre-train Model)
```bash
cd avoidance
conda activate visfly
python exps/real_world/run.py -e navigation -a SHAC -t 1
# python exps/real_world/run.py -t 1 -a PPO -e navigation -c comment
python exps/real_world/run.py -e navigation -a SHAC -t 0 -w SHAC_std_1.zip
```
If you define a new comment, replace your comment with `std` in the last command.

# Download the pre-trained weights
[Download](https://drive.google.com/file/d/1vKx2L6aIMCFVegsi_5F9uyyCsQg7XDZm/view?usp=sharing) the pre-trained weights and put it in `avoidance/exps/real_world/saved/navigation/`.
```bash
cd avoidance
conda activate visfly
python exps/real_world/run.py -e navigation -a SHAC -t 0 -w checkpoint.zip
```

# Check the test result
The test result will be saved at `exps/real_world/saved/navigation/test/checkpoint/`. You can adjust the content of saved figure and video in `exps/test.py` and rendering configuration at `eval_env.scene_kwargs.render_settings` in `exps/real_world/env_cfgs/navigation.yaml`.

# Finetune the policy
If you wanna to finetune the policy, create another env or directly modify NavigationEnv.get_reward(). Then
```bash
python exps/real_world/run.py  -e navigation  -a SHAC -t 1 -w checkpoint.zip
```
Or you wanna train the checkpoint using another algorithm:
```bash
python exps/real_world/run.py  -e navigation  -a BPTT -t 1 -w checkpoint.zip
```


