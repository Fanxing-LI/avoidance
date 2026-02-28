"""Quick compatibility test for mini experiment."""
import os
import sys
sys.path.append(os.getcwd())

from VisFly.utils.common import load_yaml_config
from envs.RealNavigationEnv import RealNavigationEnv
from algorithms.BPTT_series.SHAC import SHAC
from exps.test import Test

# Load configs
current_dir = os.path.dirname(os.path.abspath(__file__))
env_cfg_path = os.path.join(current_dir, "env_cfgs/navigation.yaml")
alg_cfg_path = os.path.join(current_dir, "alg_cfgs/navigation/SHAC.yaml")

env_config = load_yaml_config(env_cfg_path)
alg_config = load_yaml_config(alg_cfg_path)

# Use fewer scenes and agents for quick test
env_config["eval_env"]["num_scene"] = 1
env_config["eval_env"]["num_agent_per_scene"] = 1
# Keep visual from config (True for real_world compatibility)

# Weight
weights_name = "SHAC_check_checkpoint_H150r_refineRand_refineCol_remap0.2_0.3Share_treVel2_5.zip"
save_folder = os.path.join(current_dir, "saved/navigation")
weights_path = os.path.join(save_folder, weights_name)

print("Creating env...")
eval_env = RealNavigationEnv(**env_config["eval_env"])
print("Loading model...")
model = SHAC.load(weights_path, env=eval_env)
print("Creating test handle...")
test_handle = Test(model=model, save_path=save_folder + "/quick_test", name="quick_test")
print("Running test...")
test_handle.test(**alg_config["test"])
print("Test completed successfully!")
