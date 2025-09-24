
from datetime import datetime
import time
import os
from tqdm import tqdm
import argparse
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from hdrl_env import HdrlEnv
from xuance.torch.agents import PPOCLIP_Agent

def parse_args():
    parser = argparse.ArgumentParser("Example of hdrl.")
    # parser.add_argument("--env-id", type=str, default="gnnppo-srl")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":

    now = datetime.now()

    start_time = now.strftime("%H:%M:%S")
    print("Current Time =", start_time)

    parser = parse_args()
    # Get the directory of the current script to find config.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path_up = os.path.join(script_dir, "config/upper_config.yaml")
    configs_dict_up = get_configs(file_dir=config_path_up)
    configs_dict_up = recursive_dict_update(configs_dict_up, parser.__dict__)
    configs_up = argparse.Namespace(**configs_dict_up)

    # env = HdrlEnv(configs_up)

    from xuance.environment import REGISTRY_ENV
    REGISTRY_ENV[configs_up.env_name] = HdrlEnv
    set_seed(configs_up.seed)
    env = make_envs(configs_up)
    hdrl_env = env.envs[0].env
    up_agents = []
    for i in range(144):
        configs_up.log_dir = configs_up.log_dir + f"agent_{i}/"
        configs_up.model_dir = configs_up.model_dir + f"agent_{i}/"
        up_agents.append(PPOCLIP_Agent(configs_up, env))
    
    for i_episode in range(configs_up.number_episodes):
        print("---------- Episode:", i_episode+1, " ----------")
        step = []
        deliveries = []
        start = time.time()

        ''' iterate each time step try to finish routing within time_steps '''
        for t in tqdm(range(configs_up.max_steps)):
            # gen new cross-domain flows for walker-delta-net in each domain, and assign them to each up agent according to the destination node
            # to_be_injected_flows = hdrl_env.walker_delta_net.traffic_manager.tobe_assigned_flows
            # for flow in to_be_injected_flows:
            #     print(flow)
            #     up_agents[flow.destination].train([flow])
            # 
            # hdrl_env.walker_delta_net.gen_each_domain_flows
            hdrl_env.update_whole_network_state()

        hdrl_env.train_lower_level_agents()

        """ get episode return """
        hdrl_env.upload_episode_rewards()
        hdrl_env.reset()
            
        end = time.time()

        print("Time:", end - start)
    
    hdrl_env.save_lower_level_model()
    print("All episodes done!")