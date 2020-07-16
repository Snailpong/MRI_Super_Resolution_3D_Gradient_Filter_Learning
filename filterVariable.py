Qangle_t = 8
Qangle_p = 8

filter_length = 11
filter_half = filter_length // 2

# Do not edit!
Qstrength = 3
Qcoherence = 3
pixel_type = 2 ** 3

Q_total = Qangle_p * Qangle_t * Qstrength * Qcoherence
filter_volume = filter_length ** 3