import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import Jihuang._jihuang as game
import torch
import torch.nn as nn
import torch.nn.functional as F

JiHuang_dir = '/workspace/S/guojiaming/jihuang/JiHuang/python'

class JihuangEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_param="/workspace/S/guojiaming/sb3jihuang/configs/example_simple_40.prototxt", env_config="/workspace/S/guojiaming/sb3jihuang/configs/config_simple_40.prototxt", log_dir="logs", log_name="py_jihuang", log_level=0):
        self.env = game.Env(env_param, env_config, log_dir, log_name, log_level)
        # initialize the observation and action_count
        self.prev_obs = None
        self.obs = None
        self.action_count = [0, 0, 0, 0, 0, 0]
        self.reset()
        self.observation_space = gym.spaces.Box(shape=self.state.shape, low=-10000, high=10000)

        self.action_space = gym.spaces.Discrete(20)
    
    def step(self, action):
        env_action, action = get_action(action, self.struct_state)
        self.env.step([env_action])
        if self.obs is not None:
            self.prev_obs = self.obs
            self.prev_state = self.state
            self.prev_struct_state = self.struct_state
        self.action = action
        self.obs = self.env.get_agent_observe()
        # reward = self._calc_reward()
        reward = 1
        hp, bs, jk = self.obs[0][1:4]
        if hp <= 10 or bs <= 2 or jk <= 2:
            done = True
        else:
            done = False
        # info = {"action_count": self.action_count}
        #print(self.obs[0])
        action_mask = np.array(self.obs[0][7])
        if self.obs[0][7]:
            action_label = np.array(action)
        else:
            action_label = np.array(20.0) 
        info = {"action_mask": action_mask,"action_label":action_label}
        self.state, self.struct_state= new_obs2state(self.obs[0]) 
 
        return self.get_goal_obs(self.state), reward, done, info
 
    def get_goal_obs(self,state):
        return {
            "achieved_goal": self.state,
            "desired_goal": np.zeros(16),
            "observation": self.state,
        }

    def reset(self):
        self.env.reset()
        self.obs = self.env.get_agent_observe()
        self.state, self.struct_state = new_obs2state(self.obs[0]) 
        self.prev_obs, self.prev_state, self.prev_struct_state = None, None, None

        return self.get_goal_obs(self.state)


    def render(self, mode='human'):
        print(self.obs)
    
    def close(self):
        pass

#orga_atk_unit_num = 8
#orga_gtr_unit_num = 8


orga_in_atk_unit_num = 8
orga_in_gtr_unit_num = 8

orga_out_atk_unit_num = 8
orga_out_gtr_unit_num = 8

env_item_unit_num = 8
bp_item_unit_num = 16

eqbp_item_unit_num = 8
eq_item_unit_num = 5
cg_item_unit_num = 6

#env const attr:
e_map_s = 40
e_d = 50
e_m = 30
e_y = 12
landform_num = 3
weather_num = 3
action_dim = 20
target_dim = [orga_in_atk_unit_num, orga_in_gtr_unit_num, env_item_unit_num, bp_item_unit_num, eqbp_item_unit_num, eq_item_unit_num, cg_item_unit_num, bp_item_unit_num+eqbp_item_unit_num]
move_dim = 10
e_len = 14
env_attr_names = {0:'Seasn',
                  1:'Weath',
                  2:'Landf',
                  3:'Time',
                  4:'Day',
                  5:'Month',
                  6:'ShRov',
                  7:'ThRov',
                  8:'HpRov',
                  9:'TemAd',
                  10:'AtkAd',
                  11:'DefAd',
                  12:'SpdAd',
                  13:'VisAd'}
env_season_names = {0:'Sprig', 1:'Sumer', 2:'Autum', 3:'Wintr'}
env_weather_names = {0:'Sunny', 1:'Rainy'}
env_landform_names = {0:'Grass', 1:'Mount', 2:'Forst'}

#aegnt const attr:
a_len = 13
agent_attr_names = {0:'Hp',
                    1:'Sh',
                    2:'Th',
                    3:'X',
                    4:'Y',
                    5:'Bp',
                    6:'Eqbp',
                    7:'AtkRg',
                    8:'Tempt',
                    9:'ATK',
                    10:'DEF',
                    11:'Speed',
                    12:'VisRg'}
a_base_spd = 2 
a_max_spd = 3
a_atk = 60
a_def = 30
a_atk_r = 4.1 
a_v_r = 8
a_bp_s = 16
a_eqbp_s = 8

#organisms const attr:
o_len = 11
#                      spd  aggressive  collective  attackable  atk  def 
orga_attr = [np.array([  0,          0,          0,          0,   0,   0], dtype=float), #0: null
             np.array([  0,          0,          0,          0,   0,   0], dtype=float), #1: null
             np.array([  1,          0,          0,          1,   0,   0], dtype=float), #2: pig
             np.array([  2,          1,          0,          1,  60,  20], dtype=float), #3: wolf
             np.array([  0,          0,          1,          0,   0,   0], dtype=float), #4: tree
             np.array([  0,          0,          1,          0,   0,   0], dtype=float), #5: river
             np.array([  0,          0,          1,          0,   0,   0], dtype=float)] #6: mine
orga_type_table = [None,  #0: null
                   None,  #1: null
                   2,     #2: pig
                   3,     #3: wolf
                   10004, #4: tree
                   10005, #5: river
                   10006] #6: mine
orga_names = {2:'pig', 3:'wolf', 4:'tree', 5:'river', 6:'mine'}
orga_attr_names = {0:'Type', 
                   1:'X',
                   2:'Y',
                   3:'Hp',
                   4:'Dista',
                   5:'Speed',
                   6:'Aggre',
                   7:'Colle',
                   8:'Attac',
                   9:'ATK',
                   10:'DEF'} 

#item const attr
i_len = 16
#                      durl  sH_r  TH_r  HP_r  tmp_a  atk_a  def_a  spd_r  v_r_r  cons_e  eq_e  cg_e
item_attr = [np.array([   0,    0,    0,    0,     0,     0,     0,     0,     0,      0,    0,    0], dtype=float), #0: null
             np.array([   0,    0,   30,    0,     0,     0,     0,     0,     0,      1,    0,    1], dtype=float), #1: water
             np.array([   0,   30,    0,    0,     0,     0,     0,     0,     0,      1,    0,    1], dtype=float), #2: meat
             np.array([   0,    0,    0,    0,     0,     0,     0,     0,     0,      0,    0,    1], dtype=float), #3: leather
             np.array([   0,    0,    0,    0,     0,     0,     0,     0,     0,      0,    0,    1], dtype=float), #4: wood
             np.array([   0,    0,    0,    0,     0,     0,     0,     0,     0,      0,    0,    1], dtype=float), #5: stone
             np.array([  20,    0,    0,    0,    10,     0,     0,     0,     0,      0,    1,    0], dtype=float), #6: warm_stone
             np.array([  20,    0,    0,    0,     0,    40,     0,     0,     0,      0,    1,    0], dtype=float), #7: spear
             np.array([  20,    0,    0,    0,     0,     0,    20,     0,     0,      0,    1,    0], dtype=float), #8: coat
             np.array([ 100,    0,    0,    0,     0,     0,     0,     0,     8,      0,    1,    0], dtype=float), #9: torch
             np.array([  20,    0,    0,    0,     0,     0,     0,     1,     0,      0,    1,    0], dtype=float), #10: rain_shoes
             np.array([   0,    0,    0,  100,     0,     0,     0,     0,     0,      1,    0,    0], dtype=float)] #11: HP_pot
item_type_table = [30001,
                   30001, #1: water
                   30002, #2: meat
                   40003, #3: leather
                   40004, #4: wood
                   40005, #5: stone
                   80006, #6: warm_stone
                   70007, #7: spear
                   80008, #8: coat
                   90009, #9: torch
                   80010, #10: rain_shoes
                   30011] #11: HP_pot
item_names = {1:'Water',
              2:'Meat',
              3:'Leath',
              4:'Wood',
              5:'Stone',
              6:'WamSt',
              7:'Spear',
              8:'Coat',
              9:'Torch',
              10:'RaShs',
              11:'HPpot'}
item_attr_names = {0:'Type',
                   1:'X',
                   2:'Y',
                   3:'Dista',
                   4:'Durli',
                   5:'ShRov',
                   6:'ThRov',
                   7:'HpRov',
                   8:'TemAd',
                   9:'AtkAd',
                   10:'DefAd',
                   11:'SpdAd',
                   12:'VisAd',
                   13:'Consu',
                   14:'Equip',
                   15:'Craft'}

#CraftGuide           null  water  meat  leather  wood  stone
cg_table = [np.array([   0,     0,    0,       1,    1,     2], dtype=float), #6: warm stone
            np.array([   0,     0,    0,       1,    2,     1], dtype=float), #7: spear
            np.array([   0,     0,    0,       2,    1,     1], dtype=float), #8: coat
            np.array([   0,     0,    0,       0,    2,     0], dtype=float), #9: torch
            np.array([   0,     0,    0,       1,    1,     0], dtype=float), #10: rain_shoes
            np.array([   0,     1,    1,       0,    1,     0], dtype=float)] #11: HP_pot

state_dim = e_len + a_len + (orga_in_atk_unit_num+orga_in_gtr_unit_num+orga_out_atk_unit_num+orga_out_gtr_unit_num)*o_len + (env_item_unit_num+bp_item_unit_num+eqbp_item_unit_num+eq_item_unit_num+cg_item_unit_num)*i_len + (landform_num+weather_num+2)*((a_v_r*2+1)**2)
def distance2D(pos1, pos2):
    return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5
 

def new_obs2state(observation):
    #print('observation:\n', observation)
    agent_attr_num = 13
    useful_agent_attr_num = 7

    bp_capaity = 16
    eqbp_s = 8
    bp_attr_num = 2

    #eq_capaity = 2 * tool_num + 1
    eq_capaity = 3
    eq_attr_num = 2

    buff_num = 4
    buff_attr_num = 9

    # vision range
    v_r = 8
    v_d = v_r * 2 + 1
    # vision area
    v_a = v_d * v_d
    obj_attr_num = 7
    item_attr_num = 3
    wea_geo_attr_num = 4

    bp_flag_attr_num = 2

    agent_attr = torch.zeros(useful_agent_attr_num)
    # hp 饱食度 饥渴度
    agent_attr[0:3] = torch.Tensor(observation[1:4]) / 10
    # x y
    agent_attr[3:5] = torch.Tensor(observation[5:7])

    bp_begin_idx = agent_attr_num
    bp_unit = torch.Tensor(
                observation[bp_begin_idx :
                bp_begin_idx + bp_capaity * bp_attr_num]
    ).reshape(bp_capaity, bp_attr_num)
    #print('bp_unit:\n', bp_unit)
    bp_num = bp_unit[:,0].ne(0).sum()

    eq_begin_idx = bp_begin_idx + bp_capaity * bp_attr_num
    eq_unit = torch.Tensor(
                    observation[eq_begin_idx :
                    eq_begin_idx + eq_capaity * eq_attr_num]
    ).reshape(eq_capaity, eq_attr_num)
    #print('eq_unit:\n', eq_unit)

    #暂时没有装备
    agent_attr[5] = bp_capaity - bp_num
    agent_attr[6] = 0
    #print('agent_attr:\n', agent_attr)

    buff_begin_idx = eq_begin_idx + eq_capaity * eq_attr_num
    buff_unit = torch.Tensor(
                    observation[buff_begin_idx :
                    buff_begin_idx + buff_num * buff_attr_num]
    ).reshape(buff_num, buff_attr_num)
    #print('buff_unit:\n', buff_unit)

    obj_begin_idx = buff_begin_idx + buff_num * buff_attr_num
    obj_info = observation[obj_begin_idx : obj_begin_idx + v_a * obj_attr_num]
    #(attr v_X v_Y)
    obj_unit = torch.Tensor(obj_info).reshape(v_d, v_d, obj_attr_num).transpose(1,2).transpose(0,1)
    #只保留类型、坐标xy和hp
    obj_unit = obj_unit[:4,:,:]
    # pig的hp除100
    obj_unit[3, :, :] = obj_unit[3,:,:] / 100
    # 为了能够embedding
    obj_unit[0, :, :] = obj_unit[0,:,:] % 4
    #print('obj_unit:\n', obj_unit)
    pig_list = []
    pig_num = 0
    river_list = []
    river_num = 0
    #set organisms list
    pig_dt = np.dtype([('idx', int), ('x', float), ('y', float), ('hp', float), ('distance', float)])
    river_dt = np.dtype([ ('idx', int),('x', float), ('y', float), ('distance', float)])
    for i in range(v_a):
        if obj_info[i * obj_attr_num] == 2:
            pig_list.append((pig_num,
                            obj_info[i * obj_attr_num + 1],
                            obj_info[i * obj_attr_num + 2],
                            obj_info[i * obj_attr_num + 3] / 100,
                            distance2D(obj_info[i * obj_attr_num + 1: i * obj_attr_num + 3], observation[5:7])
                            ))
            pig_num += 1
        if obj_info[i * obj_attr_num] == 10005:
            river_list.append((river_num,
                            obj_info[i * obj_attr_num + 1],
                            obj_info[i * obj_attr_num + 2],
                            distance2D(obj_info[i * obj_attr_num + 1: i * obj_attr_num + 3], observation[5:7])
                            ))
            river_num += 1
    pig_list = np.sort(np.array(pig_list, dtype=pig_dt), order='distance')
    river_list = np.sort(np.array(river_list, dtype=river_dt), order='distance')

    item_begin_idx = obj_begin_idx + v_a * obj_attr_num
    item_info = observation[item_begin_idx : item_begin_idx + v_a * item_attr_num]
    #(attr X Y)
    item_unit = torch.Tensor(item_info).reshape(v_d, v_d, item_attr_num).transpose(1,2).transpose(0,1)
    #为了能够embedding
    item_unit[0,:,:] = item_unit[0,:,:] % 4
    #print('item_unit:\n', item_unit)
    water_list = []
    water_num = 0
    food_list = []
    food_num = 0
    env_item_dt = np.dtype([('idx', int), ('x', float), ('y', float), ('distance', float)])
    for i in range(v_a):
        if item_info[i * item_attr_num] == 30001:
            water_list.append((water_num,
                            item_info[i * item_attr_num + 1],
                            item_info[i * item_attr_num + 2],
                            distance2D(item_info[i * item_attr_num + 1 : i * item_attr_num + 3], observation[5:7])
                            ))
            water_num += 1
        if item_info[i * item_attr_num] == 30002:
            food_list.append((food_num,
                            item_info[i * item_attr_num + 1],
                            item_info[i * item_attr_num + 2],
                            distance2D(item_info[i * item_attr_num + 1 : i * item_attr_num + 3], observation[5:7])
                            ))
            food_num += 1
    water_list = np.sort(np.array(water_list, dtype=env_item_dt), order='distance')
    food_list = np.sort(np.array(food_list, dtype=env_item_dt), order='distance')

    wea_geo_begin_idx = item_begin_idx + v_a * item_attr_num
    assert(wea_geo_begin_idx + v_a * wea_geo_attr_num == len(observation))

    # bp_flag为背包内水和食物的数量
    bp_flag = torch.zeros(bp_flag_attr_num)
    bp_item_unit_list = np.zeros((bp_capaity, i_len))
    bp_num = 0
    for i in range(bp_capaity):
        if bp_unit[i, 0] == 0: break
        item_type = int(bp_unit[i, 0] % 10000)
        if item_type == 1:
            bp_flag[0] += 1
        if item_type == 2:
            bp_flag[1] += 1
        bp_item_unit_list[i][0] = item_type
        #TODO:check bp info length
        bp_item_unit_list[i][4:] = item_attr[item_type]
    #print('bp_flag:\n', bp_flag)

    #暂时没有torch
    torch_flag = np.zeros(3)

    # print('agent_attr',agent_attr)
    # print('bp_flag', bp_flag, bp_flag.shape)
    # print('item_unit', item_unit.shape)

    # embed (d, v, v) to (v, v, 4 + (d-1)):
    def embed(state):
      shape = state.shape
      d0 = state[0:1]
      dr = state[1:]
      # d0_embed = F.one_hot(d0.long(), 4)
      d0_embed = torch.zeros(4, *shape[1:]).scatter_(0, d0.long(), 1)
      return torch.cat([d0_embed, dr], 0)

    state = torch.cat([agent_attr.contiguous().view(-1),
                        bp_flag.contiguous().view(-1),
                        embed(obj_unit.contiguous()).view(-1),
                        embed(item_unit.contiguous()).view(-1)]
                        ).numpy()    


    struct_state = [agent_attr.numpy(),
                    pig_list, river_list, water_list, food_list,
                    bp_item_unit_list,
                    torch_flag
                    ]
    return state, struct_state


def get_action(action, struct_state):
    move_offset=[[-2,0],
                 [-1,-1],[-1,0],[-1,1],
                 [0,-2],[0,-1],[0,1],[0,2],
                 [1,-1],[1,0],[1,1],
                 [2,0]]
    
    action_idx = int(action)
    action_idx+=1
    if action_idx == 0:
        env_action = [0, 0, 0, 0]
    elif action_idx ==1:
        #sel_target_type = int(struct_state[4][0][0])
        #assert(sel_target_type != 0)
        if struct_state[1].shape[0] == 0:
            #视野内无pig
            target_x = -1
            target_y = -1
        else:
            target_x = int(struct_state[1][0][1])
            target_y = int(struct_state[1][0][2])
        env_action = [0, 1, target_x, target_y]
    elif action_idx == 2:
        #sel_target_type = int(struct_state[5][0][0])
        #assert(sel_target_type != 0)
        if struct_state[2].shape[0] == 0:
            #视野内无river
            target_x = -1
            target_y = -1
        else:
            target_x = int(struct_state[2][0][1])
            target_y = int(struct_state[2][0][2])
        env_action = [0, 2, target_x, target_y]
    elif action_idx == 3:
        env_action = [0, 3, 30001, 1]
    elif action_idx == 4:
        env_action = [0, 3, 30002, 1]
    elif action_idx == 5:
        env_action = [0, 4, 30001, 1]
    elif action_idx == 6:
        env_action = [0, 4, 30002, 1]
    elif action_idx == 7:
        env_action = [0, 5, 90009, 1]
    elif action_idx == 8:
        env_action = [0, 6, 90009, 1]
    elif action_idx > 8:
        env_action = [0, 8, move_offset[action_idx-9][0],move_offset[action_idx-9][1]]
    else:
        assert(0)
    return env_action, action

def get_return(ob, pre_ob,action):
    done = ob[2] == 100 and ob[3] == 100
    reward = -10 if done else 0
    #if ob[7]==0:
    #    reward-=0.2
    #else:
    #    reward+=0.2
    if ob[3]-pre_ob[3]>0 and not done:
        #print('successfully')
        #print(ob[7],action)
        reward+=(ob[3]-pre_ob[3])/30*5
    if ob[2]-pre_ob[2]>0 and not done:
        reward+=(ob[2]-pre_ob[2])/30*5
        #print(ob[7],action)
    
    return reward, done,ob[7]

def get_result(ob):
    return ob[7]







