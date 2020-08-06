TRAIN_GLOB = './train/*.nii.gz'
TEST_GLOB = "./test/*.nii.gz"
RESULT_DIR = "./result/"

QVF_FILE = './arrays/QVF'
H_FILE = './arrays/h'

Q_ANGLE_T = 8
Q_ANGLE_P = 8

FILTER_LEN = 11
FILTER_HALF = FILTER_LEN // 2

GRAD_LEN = 9
GRAD_HALF = GRAD_LEN // 2

Q_STRENGTH = 3      # Do not edit!
Q_COHERENCE = 3     # Do not edit!

FACTOR = 2
PIXEL_TYPE = FACTOR ** 3

Q_TOTAL = Q_ANGLE_P * Q_ANGLE_T * Q_STRENGTH * Q_COHERENCE
FILTER_VOL = FILTER_LEN ** 3

TRAIN_DIV = 3
SHARPEN = 'False'
BLEND_THRESHOLD = 10

LR_TYPE = 'interpolation'
FEATURE_TYPE = 'strength_coherence'
TRAIN_FILE_MAX = 0


def argument_parse():
    import argparse
    import sys

    global QVF_FILE, H_FILE, Q_ANGLE_T, Q_ANGLE_P, GRAD_LEN, FILTER_LEN, FILTER_HALF
    global Q_STRENGTH, Q_COHERENCE, FACTOR, PIXEL_TYPE, Q_TOTAL, FILTER_VOL, TRAIN_DIV
    global SHARPEN, BLEND_THRESHOLD, LR_TYPE, FEATURE_TYPE, TRAIN_FILE_MAX

    parser = argparse.ArgumentParser()

    parser.add_argument('--qvf_file', required=False, default=QVF_FILE)
    parser.add_argument('--h_file', required=False, default=H_FILE)
    parser.add_argument('--q_angle_t', required=False, default=Q_ANGLE_T)
    parser.add_argument('--q_angle_p', required=False, default=Q_ANGLE_P)
    parser.add_argument('--filter_len', required=False, default=FILTER_LEN)
    parser.add_argument('--grad_len', required=False, default=GRAD_LEN)
    parser.add_argument('--factor', required=False, default=FACTOR)
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
    assert args.feature_type in ['strength_coherence', 'strength_fa', 'trace_coherence', 'trace_fa']
    assert int(args.train_file_max) >= 0

    QVF_FILE = args.qvf_file
    H_FILE = args.h_file
    Q_ANGLE_T = int(args.q_angle_t)
    Q_ANGLE_P = int(args.q_angle_t)
    FILTER_LEN = int(args.filter_len)
    FILTER_HALF = FILTER_LEN // 2
    GRAD_LEN = int(args.grad_len)
    GRAD_HALF = GRAD_LEN // 2
    FACTOR = int(args.factor)
    PIXEL_TYPE = FACTOR ** 3
    Q_TOTAL = Q_ANGLE_P * Q_ANGLE_T * Q_STRENGTH * Q_COHERENCE
    FILTER_VOL = FILTER_LEN ** 3
    TRAIN_DIV = int(args.train_div)
    SHARPEN = (args.sharpen == 'True')
    BLEND_THRESHOLD = int(args.blend_threshold)
    LR_TYPE = args.lr_type
    FEATURE_TYPE = args.feature_type
    TRAIN_FILE_MAX = int(args.train_file_max)