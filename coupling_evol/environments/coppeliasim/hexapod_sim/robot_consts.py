import numpy as np

cor_1_1 = 0.25    # default position for front and hind legs coxa
cor_1_2 = 0.0     # default position for middle legs coxa
cor_2 = 0.5       # default position for femurs
cor_3 = 0.2       # default position for tibia

CORRECTION = np.array([-cor_1_1, -cor_1_1, cor_1_1, cor_1_1, cor_1_2, cor_1_2,
                        cor_2, cor_2, cor_2, cor_2, cor_2, cor_2,
                        cor_3, cor_3, cor_3, cor_3, cor_3, cor_3])


TIME_FRAME = 10#45 #the length of the simulation step and real robot control frame [ms]

# SERVOS_BASE = np.array([0, 0, -np.pi/6, -np.pi/6, -np.pi/3, -np.pi/3,
#                         0, 0, -np.pi/6, -np.pi/6, -np.pi/3, -np.pi/3,
#                         0, 0, -np.pi/6, -np.pi/6, -np.pi/3, -np.pi/3])

ct_pos = 0
cf_pos = 0
ft_pos = -np.pi/2

SERVOS_BASE = np.array([ct_pos, ct_pos, cf_pos, cf_pos, ft_pos, ft_pos,
                        ct_pos, ct_pos, cf_pos, cf_pos, ft_pos, ft_pos,
                        ct_pos, ct_pos, cf_pos, cf_pos, ft_pos, ft_pos])

ct_ampt_pos = 0
cf_ampt_pos = 1.0
ft_ampt_pos = 1.0
AMPT_BASE = np.array([ct_ampt_pos, ct_ampt_pos, cf_ampt_pos, cf_ampt_pos, ft_ampt_pos, ft_ampt_pos,
                      ct_ampt_pos, ct_ampt_pos, cf_ampt_pos, cf_ampt_pos, ft_ampt_pos, ft_ampt_pos,
                      ct_ampt_pos, ct_ampt_pos, cf_ampt_pos, cf_ampt_pos, ft_ampt_pos, ft_ampt_pos])

MAPPING = np.array([0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 16, 17], dtype=int)

POS_NEG = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
# POS_NEG = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]) * -1

DEFAULT_POS = (SERVOS_BASE + CORRECTION[MAPPING]) * POS_NEG
