"""This module was built to create the necessary objects and functions to attempt to build
a neuro evolution algorithm. In particular this implements the Evolutionary portion more"""

import math
import random
import numpy as np
import neural_net_framework as nnfw


class Population:
    """Class representing a population of critters requires inputs of population size(int), 
    environment(Env), and goal(list). goal must be a list of size 2 with ints that are within the screensize of env
    """

    def __init__(self, size, environment, goal):
        self.goal = goal
        self.arena = environment
        self.pop = self.gen_pop(size)
        self.size = size
        self.average_fitness = None
        self.generation = 0

    def gen_pop(self,size):
        """generates the population"""
        pop = []
        for _ in range(size):
            pop.append(Critter(0,0,self.arena,self.goal))
        return pop

    def set_goal(self,goal):
        """Allows the option to adjust goal location"""
        self.goal=goal
        for critter in self.pop:
            critter.goal = goal

    def tourney_parent_selection(self, k):
        """implements tournament selection of parents k represents tourney grouping size"""
        current_tourney = random.sample(self.pop, k=k)
        holder =sorted(current_tourney)[0]
        return holder

    def keep_top_parents_selection(self, amount_of_parents, mc):
        """completes a recombination while keeping the top preforming critters in the population"""
        fitnesses = []
        new_pop= []
        top_parents = sorted(self.pop[:amount_of_parents])
        new_pop.extend(top_parents)

        for critter in top_parents:
            fitnesses.append(critter.fitness)

        for _ in range(round((self.size-amount_of_parents)/2)):
            parents = random.choices(top_parents,fitnesses, k=2)
            #children = self.layer_swap(parents[0], parents[1])
            children = self.sp_crossover(parents[0],parents[1])
            #children = self.neuron_swap(parents[0], parents[1])
            new_critters = [Critter(1,1,self.arena,self.goal),Critter(1,1,self.arena,self.goal)]
            for n in range(2):
                new_critters[n].genome=children[f"child{n}"]
                self.mutation(new_critters[n], mc= mc)
            new_pop.extend(new_critters)

        return new_pop

    def partial_pop_mutation(self, fraction_to_keep):
        """forces mutation on a fraction of the population but does not change the top performers"""
        partial_pop_size = round(len(self.pop)*fraction_to_keep)
        new_pop= self.pop[:partial_pop_size]
        old_pop = self.pop[partial_pop_size:]
        for critter in old_pop:
            mutated_critter = self.mutation(critter, mc = 1, upperbound=2.0, lowerbound=-2.0)
            new_pop.append(mutated_critter)
        self.pop = new_pop

    def mutation(self, critter, mc = .2, upperbound = 1.0, lowerbound = -1.0):
        """mutates a critter a random node in each layer using Mutation Chance(mc) as the probability and the
        upper and lower bounds are how much a single mutation can change the float value"""
        if random.randint(0,100) < mc*100:
            for layer in range(len(critter.genome[0])):
                genome = critter.genome[0][layer]

                g_pull_index = random.randint(0,len(genome)-1)

                for weight in range(len(genome[g_pull_index])):
                    current_weight = critter.genome[0][layer][g_pull_index][weight]
                    critter.genome[0][layer][g_pull_index][weight]= current_weight + random.uniform(lowerbound,upperbound)

                #genome[g_pull_index] = g_holder
                #bias[b_pull_index] = b_holder
            return critter
        else:
            return critter

    def neuron_swap(self, parent1, parent2):
        """Randomly selects a neuron in the neural net and swaps all weights"""
        children = {"child0": None, "child1": None}
        random_layer = random.randrange(len(parent1.genome[0]))
        random_neuron = random.randrange(len(parent1.genome[0][random_layer]))

        holder = parent1.genome[0][random_layer][random_neuron]
        parent1.genome[0][random_layer][random_neuron] = parent2.genome[0][random_layer][random_neuron]
        parent2.genome[0][random_layer][random_neuron] = holder

        children["child0"] = [parent1.genome[0],parent1.genome[1]]
        children["child1"] = [parent2.genome[0],parent2.genome[1]]
        return children

    def layer_swap(self, parent1, parent2):
        """Swaps a layer between two critters"""
        children = {"child0": None, "child1": None}
        crossing_layer = random.randrange(len(parent1.genome[0]))

        holder = parent1.genome[0][crossing_layer]
        parent1.genome[0][crossing_layer] = parent2.genome[0][crossing_layer]
        parent2.genome[0][crossing_layer] = holder

        children["child0"] = [parent1.genome[0], parent1.genome[1]]
        children["child1"] = [parent2.genome[0], parent2.genome[1]]

        return children

    def sp_crossover(self, parent1, parent2):
        """Goes through each layer and swaps neurons based on a random crossing point"""
        weights = [[],[]]
        children = {"child0":None, "child1": None}
        #crossover weights
        for layer in range(len(parent1.genome[0])):
            crossing_point = random.randrange(0,len(parent1.genome[0][layer]))
            child0 = np.concatenate((parent1.genome[0][layer][:crossing_point],
                                     parent2.genome[0][layer][crossing_point:]), axis= 0)
            child1 = np.concatenate((parent2.genome[0][layer][:crossing_point],
                                     parent1.genome[0][layer][crossing_point:]), axis= 0)
            weights[0].append(child0)
            weights[1].append(child1)
        #crossover biases
        b_crossing_point = random.randrange(0,len(parent1.genome[1]))
        bias0 = parent1.genome[1][:b_crossing_point] + parent2.genome[1][b_crossing_point:]
        bias1 = parent2.genome[1][:b_crossing_point] + parent1.genome[1][b_crossing_point:]

        children["child0"] = [weights[0],bias0]
        children["child1"] = [weights[1],bias1]
        return children

    def selection_recombo(self, parents_to_keep, mc):
        """kinda useless used to represent selection and recombination"""
        self.pop = self.keep_top_parents_selection(parents_to_keep, mc= mc)

    def update_fitness(self):
        """Updates all critters fitnesses in a given population based on there distance to the goal"""
        average_fitness = []
        for critter in self.pop:
            critter.fitness = critter.gen_fitness(critter.goal, must_touch_goal= False)
            average_fitness.append(critter.fitness)
        self.average_fitness = sum(average_fitness)/len(self.pop)
        self.pop = sorted(self.pop)

    def multi_update_fitness(self, attempts):
        """Updates all critters fitnesses in a given population based on there distance to the goal"""
        average_fitness = []
        for critter in self.pop:
            critter.fitness = sum(critter.multi_attempt_fitness)/attempts
            average_fitness.append(critter.fitness)
        self.average_fitness = sum(average_fitness)/len(self.pop)
        self.pop = sorted(self.pop)

class Env():
    """Builds the environment that the population exists in,
    currently just a simple grid"""
    def __init__(self, screen_width, screen_height, ep_length):
        self.width = screen_width
        self.height = screen_height
        self.farthest_distance = math.sqrt(((self.width/2)**2) + ((self.height/2)**2))
        self.pop = None
        self.ep_length = ep_length #how many frames to run for

    def set_population(self, population):
        """Sets the population for the environment"""
        self.pop = population

    def multi_attempt_step(self, neural_network, attempts):
        """Experimental attempt at having each critter try at a random goal (attempts) times for generalization
        works by each critter fully trying 3 episodes instead of each critter making one move like in the step method"""
        for critter in self.pop.pop:
            critter.multi_attempt_fitness = []
            for _ in range(self.ep_length):
                for _ in range(attempts):
                    neural_network.set_weights(critter.genome[0])
                    neural_network.set_inputs(critter.position + critter.goal)
                    direction = neural_network.predict()
                    critter.move(1, direction)
            critter.multi_attempt_fitness.append(critter.gen_fitness(critter.goal, must_touch_goal= False))
        self.pop.multi_update_fitness(attempts)

    def step(self, neural_network):
        """Runs a single episode allowing each individual to develop there fitness"""
        for _ in range(self.ep_length):
            for critter in self.pop.pop:
                neural_network.set_weights(critter.genome[0])
                neural_network.set_inputs(critter.position + critter.goal)
                direction = neural_network.predict()
                critter.move(1, direction)
        self.pop.update_fitness()

    def reset(self):
        """Resets all critters in the environment"""
        for critter in self.pop.pop:
            critter.position = [0,0]

class Critter:
    """Critter class creates a single critter. Inputs are for a position an environment and goal location"""
    def __init__(self, x, y, environment, goal):
        self.position = [x,y]
        self.goal = goal
        self.genome = self.gen_genome(4, 3, 4, 4) #MUST BE SAME SHAPE AS NEURAL NET
        self.arena = environment
        self.fitness = self.gen_fitness(self.goal, must_touch_goal=False)
        self.multi_attempt_fitness = []

    def __lt__(self, other):
        """dunder method that allows the use of sort based on fitness"""
        return self.fitness > other.fitness

    def gen_genome(self,num_of_inputs, num_of_layers, num_of_outputs, size_of_dense_layer):
        """Generates a genome that is passes into a neural network"""
        #num_of_inputs = 4, size_of_dense_layer = 8, num_of_outputs = 4
        genome_array = [0.10 * np.random.randn(num_of_inputs, size_of_dense_layer)]#starting layer initialized
        #bias_array = [np.random.uniform(-10,10, (1,size_of_dense_layer))]
        bias_array = [np.zeros((1,size_of_dense_layer))]
        #for dense layers
        for _ in range(num_of_layers - 3):
            genome_array.append(0.10 * np.random.randn(size_of_dense_layer, size_of_dense_layer))
            #bias_array.append(np.random.uniform(-10,10, (1,size_of_dense_layer)))
            bias_array = [np.zeros((1,size_of_dense_layer))]
        #for output layer
        genome_array.append(0.10 * np.random.randn(size_of_dense_layer, num_of_outputs))
        #bias_array.append(np.random.uniform(-10,10, (1,num_of_outputs)))
        bias_array = [np.zeros((1,size_of_dense_layer))]

        genome_array.append(0.10 * np.random.randn(num_of_outputs, num_of_outputs))
        #bias_array.append(np.random.uniform(-10,10, (1,num_of_outputs)))
        bias_array = [np.zeros((1,size_of_dense_layer))]
        return [genome_array, bias_array]

    def _distance_to(self, pos_x2, pos_y2):
        """generates the distance while accounting for the screen wrapping around"""
        dx = abs(self.position[0] - pos_x2)
        dy = abs(self.position[1] - pos_y2)
        if dx>self.arena.height/2:
            dx = self.arena.height - dx
        if dy>self.arena.width/2:
            dy = self.arena.width - dy
        return math.sqrt((dx**2)+dy**2)

    def gen_fitness(self, goal, must_touch_goal = False):
        """Generates fitness 1 being perfect"""
        if must_touch_goal is True: #experimental on only being rewarded for finding the location
            if self._distance_to(goal[0],goal[1]) <= 3:# within 3 points
                fitness = 1
            else:
                fitness=0.01
        else:
            fitness = (self.arena.farthest_distance-self._distance_to(goal[0],goal[1]))/self.arena.farthest_distance
            if fitness <= 0:
                fitness = .001
        return fitness

    def move(self, step, prediction):
        """Calculates a move based on a softmax output"""
        m = max(prediction)
        i = m.argmax()
        if i == 0:
            self.position[1] += step#North
            if self.position[1] > self.arena.height:
                self.position[1] = self.position[1] - self.arena.height -1#set for screen wrapping
        elif i == 1:
            self.position[0] += step#East
            if self.position[0] > self.arena.width:
                self.position[0] = self.position[0] - self.arena.width -1
        elif i == 2:
            self.position[1] -= step#South
            if self.position[1] < 0:
                self.position[1] = self.position[1] + self.arena.height+1
        elif i == 3:
            self.position[0] -= step#West
            if self.position[0] < 0:
                self.position[0] = self.position[0] + self.arena.width+1

def gen_neural_net(num_of_inputs, dense_layer_size, num_of_outputs, num_of_layers):
    """creates a neural net that must be the same shape as the genome"""
    layers = [nnfw.LayerDense(num_of_inputs, dense_layer_size, activation= "ReLU")]
    for _ in range(num_of_layers-3):
        layers.append(nnfw.LayerDense(dense_layer_size, dense_layer_size, activation= "ReLU"))
    #need this so that softmax correctly looks between 4 options
    layers.append(nnfw.LayerDense(dense_layer_size, num_of_outputs, activation= "ReLU"))
    layers.append(nnfw.LayerDense(num_of_outputs, num_of_outputs, activation= "Softmax"))

    network_model = nnfw.NNModel(layers)
    return network_model



if __name__ == "__main__":
    arena = Env(1280,720, 20)
    test = Critter(1, 1,arena, goal= [640,360])
    test2 = Critter(1, 1,arena, goal= [640,360])
    print(test.genome[0][0].shape)
    model = gen_neural_net(4,4,4,3)

    testPop = Population(2, environment= arena, goal= [640, 360])

    for i in range(len(test.genome[0])):
        if test.genome[0][i].shape == model.layers[i].weights.shape:
            print("weights are the same shape")

    print(test.position+[640,360])
    model.set_inputs(test.position+[640,360])
    print(model.inputs)
    print("prediction with inputs",model.predict())
    test.gen_fitness([640,360])
    print("starting fitness:",test.fitness)

    testPop.layer_swap(test, test2)

    array = np.array([1,5,3,2])
    print(array)
    print(array +random.uniform(1,5))
