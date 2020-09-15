import numpy as np
import os
import pickle
from math import ceil

import filter_constant as C


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.nii.gz'):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def ask_save_qv(Q, V, finished_files):
    try:
        a = input_timer("\r Enter to save >> ", 10)
        save_qv(Q, V, finished_files)
    except TimeoutError as e:
        pass


def save_qv(Q, V, finished_files):
    print('\rSaving QVF...', end='', flush=True)
    np.savez('{}QVF_{}'.format(C.ARRAY_DIR, C.R), Q=Q, V=V, finished_files=np.array(finished_files))


def save_boundary(stre, cohe):
    np.savez('{}boundary_{}'.format(C.ARRAY_DIR, C.R), stre=stre, cohe=cohe)


def load_upscale_array():
    h = np.load('{}h_{}.npy'.format(C.ARRAY_DIR, C.R))
    h = np.array(h, dtype=np.float32)

    boundary = np.load('{}boundary_{}.npz'.format(C.ARRAY_DIR, C.R))
    stre = boundary['cohe']
    cohe = boundary['stre']
    boundary.close()

    return h, stre, cohe


def init_buckets():
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS


def load_files():
    if os.path.isfile('{}QVF_{}.npz'.format(C.ARRAY_DIR, C.R)):
        print('Loading QVF...', end=' ', flush=True)
        QVF = np.load('{}QVF_{}.npz'.format(C.ARRAY_DIR, C.R))
        Q = QVF['Q']
        V = QVF['V']
        finished_files = QVF['finished_files'].tolist()
        QVF.close()
        print('Done', flush=True)
    else:
        Q = np.zeros((C.Q_TOTAL, C.FILTER_VOL, C.FILTER_VOL))
        V = np.zeros((C.Q_TOTAL, C.FILTER_VOL))
        finished_files = []

    return Q, V, finished_files

def chunk(lst, size):
    return list(map(
                   lambda x: lst[x * size:x * size + size],
                   list(range(0, ceil(len(lst) / size)))))


# Original Code Source : https://greenfishblog.tistory.com/257
def input_timer(prompt, timeout_sec):
    
    import subprocess
    import sys
    import threading
    import locale

    class Local:
        # check if timeout occured
        _timeout_occured = False

        def on_timeout(self, process):
            self._timeout_occured = True
            process.kill()
            # clear stdin buffer (for linux)
            # when some keys hit and timeout occured before enter key press,
            # that input text passed to next input().
            # remove stdin buffer.
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except ImportError:
                # windows, just exit
                pass

        def input_timer_main(self, prompt_in, timeout_sec_in):
            # print with no new line
            print(prompt_in, end="")

            # print prompt_in immediately
            sys.stdout.flush()

            # new python input process create.
            # and print it for pass stdout
            cmd = [sys.executable, '-c', 'print(input())']
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                timer_proc = threading.Timer(timeout_sec_in, self.on_timeout, [proc])
                try:
                    # timer set
                    timer_proc.start()
                    stdout, stderr = proc.communicate()

                    # get stdout and trim new line character
                    result = stdout.decode(locale.getpreferredencoding()).strip("\r\n")
                finally:
                    # timeout clear
                    timer_proc.cancel()

            # timeout check
            if self._timeout_occured is True:
                # move the cursor to next line
                #print("")
                raise TimeoutError
            return result

    t = Local()
    return t.input_timer_main(prompt, timeout_sec)