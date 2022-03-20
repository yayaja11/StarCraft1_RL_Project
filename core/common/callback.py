from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np

from keras import __version__ as KERAS_VERSION
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList  # 케라스 라이브러리에서 가져오는 콜백
from keras.utils.generic_utils import Progbar

class Callback(KerasCallback):
    def __init__(self, agent = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent

        def _set_env(self, env):
            self.env = env

        def on_episode_begin(self, episode, logs={}):
            """매 에피소드 시작마다 호출되는 곳"""
            pass

        def on_episode_end(self, episode, logs={}):
            """에피소드 종료마다 호출되는 곳"""
            pass

        def on_step_begin(self, step, logs={}):
            """매 step 시작마다 호출되는 곳"""
            pass

        def on_step_end(self, step, logs={}):
            """매 step 종료마다 호출되는 곳"""
            pass

        def on_action_start(self, action, logs={})
            """매 행동 시작마다 호출되는 곳"""
            pass

        def on_action_end(self, action, logs={})
            """매 행동 종료마다 호출되는 곳"""
            pass

class CallbackList(KerasCallbackList):
    def _set_env(self,env):
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)

    def on_episode_begin(self, episode, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_episode_end', None))
                callback.on_episode_end(episode, logs=logs)
            else:
                from keras.callbacks import TensorBoard
                if type(callback) is TensorBoard:
                    logs.pop('info')
                    callback.on_epoch_end(episode, logs=logs)
                else:
                    callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_episode_begin(step, logs=logs)
            else:
                callback.on_epoch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                from tensorflow.python.keras.callbacks import TensorBoard
                if type(callback) is TensorBoard:
                    scalar_logs = {}
                    for k, v in logs.items():
                        if type(v) in [np.int32, np.int16, np.int64, np.float32, np.float64]:
                            scalar_logs[k] = v
                    callback.on_batch_end(step, logs=scalar_logs)
                else:
                    callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)



