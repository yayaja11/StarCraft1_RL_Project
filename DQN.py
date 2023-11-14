import bunker_map as bm
from keras.models import Model
from keras.layers import Dense
import tensorflow as tf


class DQN(Model):
    action_n = len(bm.STATE_BUNKER)
    def __init__(self, action_n):
        super(DQN, self).__init__()

        self.h0 = Dense(128, activation='relu')
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(action_n, activation='linear')

    def call(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)

        q = self.q(x)
        return q

class Dueling_DQN(Model):

    def __init__(self, action_n):
        super(Dueling_DQN, self).__init__()

        self.h0 = Dense(128, activation='relu')
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')

        self.fc1_v = Dense(16)
        self.fc1_adv = Dense(16)
        self.fc2_v = Dense(1)
        self.fc2_adv = Dense(action_n)


    def call(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        adv = self.fc1_adv(x)
        v = self.fc1_v(x)

        adv = self.fc2_adv(adv)
        v = self.fc2_v(v)


        q = (v + (adv - tf.math.reduce_mean(adv, axis = 1, keepdims= True)))
        return q


