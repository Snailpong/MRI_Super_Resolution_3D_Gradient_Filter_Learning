Q_ANGLE_T = 8
Q_ANGLE_P = 8

FILTER_LEN = 11
FILTER_HALF = FILTER_LEN // 2

GRAD_LEN = 9
GRAD_HALF = GRAD_LEN // 2

# Do not edit!
Q_STRENGTH = 3
Q_COHERENCE = 3

FACTOR = 2
PIXEL_TYPE = FACTOR ** 3

Q_TOTAL = Q_ANGLE_P * Q_ANGLE_T * Q_STRENGTH * Q_COHERENCE
FILTER_VOL = FILTER_LEN ** 3

TPB = 32
TRAIN_DIV = 3
TRAIN_STP = 15 // TRAIN_DIV

BLEND_THRESHOLD = 10