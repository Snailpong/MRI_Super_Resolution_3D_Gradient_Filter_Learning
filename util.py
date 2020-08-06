import numpy as np
import os
import pickle

import filter_constant as C


def ask_save_qv(Q, V, finished_files):
    try:
        a = input_timer("\r Enter to save >> ", 10)
        save_qv(Q, V, finished_files)
    except TimeoutError as e:
        pass

def save_qv(Q, V, finished_files):
    print('\rSaving QVF...', end='', flush=True)
    np.savez(C.QVF_FILE, Q=Q, V=V, finished_files=np.array(finished_files))

def init_buckets():
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS

def load_files():
    if os.path.isfile('{}.npz'.format(C.QVF_FILE)):
        print('Loading QVF...', end=' ', flush=True)
        QVF = np.load('{}.npz'.format(C.QVF_FILE))
        Q = QVF['Q']
        V = QVF['V']
        finished_files = QVF['finished_files'].tolist()
        QVF.close()
        print('Done', flush=True)
    else:
        Q = np.zeros((C.Q_TOTAL, C.PIXEL_TYPE, C.FILTER_VOL, C.FILTER_VOL))
        V = np.zeros((C.Q_TOTAL, C.PIXEL_TYPE, C.FILTER_VOL, 1))
        finished_files = []

    return Q, V, finished_files

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