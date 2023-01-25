import actor_call
import actor as ac
import envs.make_command_env as sc
import multiprocessing
import multi_manager
if __name__ == '__main__':
    ip = '127.0.0.1'
    port = str(11111)
    multi_manager.multi_actor(ip, port)
    multi_manager.multi_actor(ip, str(int(port) + 1))

    create = 1
    count = 0
    worker_list = []
    for worker in workers:
        worker_list.append(worker)

    if create == 1:
        one_more = 1
        count += 1
        create = 0
    worker_list[count].start()