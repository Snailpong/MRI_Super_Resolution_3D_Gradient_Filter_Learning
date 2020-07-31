import numpy as np
import os
import pickle

from filter_constant import *

def ask_save_qv(Q, V, finished_files):
    try:
        a = input_timer("\r Enter to save >> ", 10)
        save_qv(Q, V, finished_files)
    except TimeoutError as e:
        pass

def save_qv(Q, V, finished_files):
    np.save("./arrays/Q", Q)
    np.save("./arrays/V", V)
    with open('./arrays/finished_files.pkl', 'wb') as f:
        pickle.dump(finished_files, f)

def init_buckets():
    patchS = [[] for j in range(Q_TOTAL)]
    xS = [[] for j in range(Q_TOTAL)]
    return patchS, xS

def load_files():
    # Construct an empty matrix Q, V uses the corresponding LR and HR
    if os.path.isfile('./arrays/Q.npy') and os.path.isfile('./arrays/V.npy'):
        print('Importing exist arrays...', end=' ', flush=True)
        Q = np.load("./arrays/Q.npy")
        V = np.load("./arrays/V.npy")
        with open('./arrays/finished_files.pkl', 'rb') as f:
            finished_files = pickle.load(f)
        print('Done', flush=True)
        
    else:
        Q = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL, FILTER_VOL))
        V = np.zeros((Q_TOTAL, PIXEL_TYPE, FILTER_VOL, 1))
        finished_files = []

    return Q, V, finished_files

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

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
                print("")
                raise TimeoutError
            return result

    t = Local()
    return t.input_timer_main(prompt, timeout_sec)