import multiprocessing
import actor as ac
import learner as learn
import time


def actor_process(ip, port):
    ac.actor_func(ip, port)


if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        results = manager.list()
        ip = '127.0.0.1'
        port = '11111'

        workers = [multiprocessing.Process(target=actor_process, args=(ip, str(int(port) + x),)) for x in range(1)]


        for worker in workers:
            worker.start()


        for worker in workers:
            worker.join()

        # for buffer in buffers:
        #     buffer.join()
    # pool = multiprocessing.Pool(processes=3)
    # print('Results: %s' %pool.map(hi2, range(3)))