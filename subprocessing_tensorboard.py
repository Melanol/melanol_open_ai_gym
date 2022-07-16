from multiprocessing import Process
import subprocess
import time
import webbrowser


def start_tensorboard():
    cmd = 'tensorboard --logdir ./tensorboard_log'
    subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)


def open_tensorboard():
    time.sleep(2)
    url = "http://localhost:6006/ "
    webbrowser.open(url, new=0, autoraise=True)


if __name__ == '__main__':
    p1 = Process(target=start_tensorboard)
    p1.start()
    p2 = Process(target=open_tensorboard)
    p2.start()
    # p.join()
