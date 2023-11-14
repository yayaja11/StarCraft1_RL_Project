import multiprocessing
import actor as ac
import learner as learn
import time
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import ICMPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

def actor_process(ip, port, algorithm, double, dueling, q, param_q, x ):
    ac.actor_func(ip, port, algorithm, double, dueling, q, param_q, x)

def learner_process(q, param_q, double, dueling, batch_size):
    learn.learner(q, param_q, double, dueling, batch_size)

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        results = manager.list()
        ip = '127.0.0.1'
        port = '11111'
        algorithm = 'DQN'
        batch_size = 1024
        double = False
        dueling = False
        q = multiprocessing.Queue(3000)
        param_q = multiprocessing.Queue()
        workers = [multiprocessing.Process(target=actor_process, args=(ip, str(int(port) + x), algorithm, double,
                                                                       dueling, q, param_q, str(x),)) for x in range(8)]
        learners = [multiprocessing.Process(target=learner_process, args=(q, param_q, double, dueling, batch_size,)) for y in range(1)]
        # buffers = [multiprocessing.Process(target=buffer_process, args=(q,)) for z in range(1)]

        for worker in workers:
            worker.start()
        for learner in learners:
            learner.start()
        # for buffer in buffers:
        #     buffer.start()

        for worker in workers:
            worker.join()
        for learner in learners:
            learner.join()
        # for buffer in buffers:
        #     buffer.join()
    # pool = multiprocessing.Pool(processes=3)
    # print('Results: %s' %pool.map(hi2, range(3)))