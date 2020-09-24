import numpy as np

TRAIN_GLOB = './train/*.nii.gz'
TEST_GLOB = "./test/*.nii.gz"
RESULT_DIR = "./result/"

Q_ANGLE_T = 8
Q_ANGLE_P = 8

PATCH_SIZE = 11
PATCH_HALF = PATCH_SIZE // 2

GRADIENT_SIZE = 9
GRADIENT_HALF = GRADIENT_SIZE // 2

Q_LAMBDA1_SPLIT = np.array([0.0001, 0.001])
Q_TRACE_SPLIT = np.array([0.0001, 0.001])
Q_COH2_SPLIT = np.array([0.25, 0.5])
Q_FA_SPLIT = np.array([0.05, 0.1])

Q_STRENGTH = 3      # Do not edit!
Q_COHERENCE = 3     # Do not edit!

R = 2

Q_TOTAL = Q_ANGLE_P * Q_ANGLE_T * Q_STRENGTH * Q_COHERENCE
FILTER_VOL = PATCH_SIZE ** 3

TRAIN_DIV = 3
SHARPEN = 'False'
BLEND_THRESHOLD = 10

LR_TYPE = 'interpolation'
FEATURE_TYPE = 'lambda1_coh2'
TRAIN_FILE_MAX = 99999999


def argument_parse():
    import argparse
    import sys

    global Q_ANGLE_T, Q_ANGLE_P, GRADIENT_SIZE, PATCH_SIZE, PATCH_HALF
    global Q_STRENGTH, Q_COHERENCE, R, Q_TOTAL, FILTER_VOL, TRAIN_DIV
    global SHARPEN, BLEND_THRESHOLD, LR_TYPE, FEATURE_TYPE, TRAIN_FILE_MAX

    parser = argparse.ArgumentParser()

    parser.add_argument('--q_angle_t', required=False, default=Q_ANGLE_T)
    parser.add_argument('--q_angle_p', required=False, default=Q_ANGLE_P)
    parser.add_argument('--filter_len', required=False, default=PATCH_SIZE)
    parser.add_argument('--grad_len', required=False, default=GRADIENT_SIZE)
    parser.add_argument('--factor', required=False, default=R)
    parser.add_argument('--train_div', required=False, default=TRAIN_DIV)
    parser.add_argument('--sharpen', required=False, default=SHARPEN)
    parser.add_argument('--blend_threshold', required=False, default=BLEND_THRESHOLD)
    parser.add_argument('--lr_type', required=False, default=LR_TYPE)
    parser.add_argument('--feature_type', required=False, default=FEATURE_TYPE)
    parser.add_argument('--train_file_max', required=False, default=TRAIN_FILE_MAX)

    args = parser.parse_args()
    
    assert int(args.q_angle_t) > 3
    assert int(args.q_angle_p) > 3
    assert int(args.filter_len) > 2 and int(args.filter_len) % 2 == 1 
    assert int(args.grad_len) > 2 and int(args.filter_len) % 2 == 1
    assert int(args.factor) >= 2
    assert int(args.train_div) >= 1
    assert args.sharpen in ['True', 'False']
    assert 1 <= int(args.blend_threshold) <= 26
    assert args.lr_type in ['kspace', 'interpolation']
    assert args.feature_type in ['lambda1_coh2', 'lambda1_fa', 'trace_coh2', 'trace_fa']
    assert int(args.train_file_max) >= 1

    Q_ANGLE_T = int(args.q_angle_t)
    Q_ANGLE_P = int(args.q_angle_t)
    PATCH_SIZE = int(args.filter_len)
    PATCH_HALF = PATCH_SIZE // 2
    GRADIENT_SIZE = int(args.grad_len)
    GRAD_HALF = GRADIENT_SIZE // 2
    R = int(args.factor)
    Q_TOTAL = Q_ANGLE_P * Q_ANGLE_T * Q_STRENGTH * Q_COHERENCE
    FILTER_VOL = PATCH_SIZE ** 3
    TRAIN_DIV = int(args.train_div)
    SHARPEN = (args.sharpen == 'True')
    BLEND_THRESHOLD = int(args.blend_threshold)
    LR_TYPE = args.lr_type
    FEATURE_TYPE = args.feature_type
    TRAIN_FILE_MAX = int(args.train_file_max)