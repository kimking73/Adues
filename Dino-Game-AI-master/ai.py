import numpy as np
import pygame as pg
import random
import copy
import pickle
from game import Dino

RAND_INIT = 5
def rand():
    return random.uniform(0, RAND_INIT) # 2개의 실수 사이 값 1개 리턴(실수)

def np_rand(arr, row ,col):
    for i in range(row):
        for j in range(col):
            arr[i][j] = rand() # 

def ReLU(x):
    return np.maximum(0, x) 

def identity_function(x):
    return x

class Instance:
    def __init__(self):
        self.dino = Dino(44,47)
        self.init_network()
        self.X = np.array([1.0, 1.0])
        self.action = 3

    def init_network(self):
        # TODO : find the appropriate number of nodes
        self.network = {}

        W1 = np.zeros((2, 3))
        np_rand(W1, 2, 3)

        b1 = np.zeros((1, 3))
        np_rand(b1, 1, 3)

        self.network['W1'] = W1
        self.network['b1'] = b1
        

    def get_enemy_pos(self, cacti, ptreas):
        front_ob = pg.Rect(600, 100, 40, 40)
        front_c = pg.Rect(600, 100, 40, 40)
        for c in cacti:
            if c.rect.x < front_c.x and c.rect.x > 70:
                front_c = c.rect

        front_p = pg.Rect(600, 100, 40, 40)
        for p in ptreas:
            if p.rect.x < front_p.x and p.rect.x > 70:
                front_p = p.rect

        if front_p.x > front_c.x:
            front_ob = front_c
        else:
            front_ob = front_p
        
        #print(front_ob)
                
        self.X = np.array([front_ob.x, front_ob.y])

    def print_network(self):
        for key in self.network.keys():
            print(key, "\n", self.network[key])
        print()
        
    def forward(self, c, p):
        self.get_enemy_pos(c, p)

        a1 = np.dot(self.X, self.network['W1']) + self.network['b1']
        z1 = ReLU(a1)
        
        y = z1
        
        self.action = np.argmax(y)

    def copy(self, new):
        new.network = copy.deepcopy(self.network)
        

class Generation:
    def __init__(self, n):
        self.generation = 1
        self.num = n
        self.instance = []
        self.prev_high = 0
        self.T = 100
        self.gene_score = [0]
        self.create_instance()
        
        f = open('output.txt','w')
        f.close()
    
    def create_instance(self):
        for i in range(self.num):
            self.instance.append(Instance())

    def generation_end(self):
        for i in range(self.num):
            if self.instance[i].dino.isDead == False:
                return False
        return True

    def save_score(self):
        self.gene_score = []
        for i in range(self.num):
            self.gene_score.append(self.instance[i].dino.score)

    def info(self):
        # print()
        # print("============================================")
        print("\t\tGeneration "+ str(self.generation),"\t\tBest record "+ str(max(self.gene_score)))

        # print("============================================")
        # print()
        if self.prev_high < max(self.gene_score):
            if self.T > 1:
                self.T -= 1
            self.prev_high = max(self.gene_score)

        f = open('output.txt','a')
        data = str(self.generation) + " " + str(max(self.gene_score)) + "\n"
        f.write(data)

    def selection(self):
        for i in range(int(self.num * 0.2)):
            index = self.gene_score.index(max(self.gene_score))
            tmp = Instance()
            self.instance[index].copy(tmp)
            self.new_instance.append(tmp)

            del self.instance[index]
            del self.gene_score[index]

    def cross_over(self):
        count = 0
        for i in range(int(self.num * 0.2 / 2)):
            tmp1 = Instance()
            tmp2 = Instance()
            
            self.new_instance[count].copy(tmp1)
            self.new_instance[count + 1].copy(tmp2)
            
            for i in range(2):
                key = random.choice(list(tmp1.network.keys()))
                z = tmp1.network[key]
                tmp1.network[key] = tmp2.network[key]
                tmp2.network[key] = z
                
            count += 2
            
            self.new_instance.append(tmp1)
            self.new_instance.append(tmp2)

    def mutation(self):
        count = 0
        for i in range(int(self.num * 0.6)):
            tmp = Instance()
            self.new_instance[count].copy(tmp)
            
            for i in range(self.T):
                key = random.choice(list(tmp.network.keys()))
                target = tmp.network[key]
                x = random.randrange(0, target.shape[0])
                y = random.randrange(0, target.shape[1])

                target[x][y] = rand()
                
            self.new_instance.append(tmp)
            count += 1
            
    def new_generation(self):
        self.save_score()
        self.info()
        self.generation += 1
        self.new_instance = []
        self.selection()
        self.cross_over()
        self.mutation()
        self.instance = self.new_instance

    def get_network_list(self):
        ret = []
    
        for i in range(len(self.instance)):
            ret.append(self.instance[i].network)
        return ret

    def load_data(self, data):
        self.generation = data.g
        self.instance = []
        for i in range(len(data.data)):
            tmp = Instance()
            tmp.network = data.data[i]
            self.instance.append(tmp)


class Data:
    def __init__(self, g, data):
        self.g = g
        self.data = data
