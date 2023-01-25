id_col = []
state_info = {
    'loc0': "int8",
    'loc1': "int8",
    'loc2': "int8",
    'loc3': "int8",
    'loc4': "int8",
    'loc5': "int8",
    'loc6': "int8",
    'loc7': "int8",
    'loc8': "int8",
    'loc9': "int8",

    'loc10': "int8",
    'loc11': "int8",
    'loc12': "int8",
    'loc13': "int8",
    'loc14': "int8",
    'loc15': "int8",
    'loc16': "int8",
    'loc17': "int8",
    'loc18': "int8",
    'loc19': "int8",

    'loc20': "int8",
    'loc21': "int8",
    'loc22': "int8",
    'loc23': "int8",
    'loc24': "int8",
    'loc25': "int8",
    'loc26': "int8",
    'loc27': "int8",
    'loc28': "int8",
    'loc29': "int8",

    'loc30': "int8",
    'loc31': "int8",
    'loc32': "int8",
    'loc33': "int8",
    'loc34': "int8",
    'loc35': "int8",
    'loc36': "int8",
    'loc37': "int8",
    'loc38': "int8",
    'loc39': "int8",

    'loc40': "int8",
    'loc41': "int8",
    'loc42': "int8",
    'loc43': "int8",
    'loc44': "int8",
    'loc45': "int8",
    'loc46': "int8",
    'loc47': "int8",
    'loc48': "int8",
    'loc49': "int8",

    'loc50': "int8",
    'loc51': "int8",
    'loc52': "int8",
    'loc53': "int8",
    'loc54': "int8",
    'loc55': "int8",
    'loc56': "int8",
    'loc57': "int8",

    'life': "int8",
    'luck': "int8",
    'curr_up': "int8",
    "n_of_normal_b": "int8",
    "n_of_hero_b": "int8",
    "frame":"int8",
    "miss_action":"int8",
    "ore":"int8",
    "gas":"int8",
}
action_space = []
for i in range(117):
    action_space.append(i)
action_info = {'action':["discrete",action_space]}

metric_info = {"done": "int8",}
argument_info = {}
action_type = 'discrete'