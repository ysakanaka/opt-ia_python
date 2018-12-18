#!/usr/bin/env python3

import random
import sobol_seq
import numpy as np
import copy
import cell

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import deap.tools


class OptIA:
    MAX_GENERATION = 10000
    MAX_POP = 20
    MAX_AGE = 3
    DIMENSION = None
    LBOUNDS = None
    UBOUNDS = None
    fun = None
    evalcount = 0
    generation = 0
    best = None
    all_best = None
    all_best_generation = 0

    GENOTYPE_DUP = True
    SAIL = True

    pop = []
    clo_pop = []
    hyp_pop = []

    original_coordinates = []
    original_vals = []
    searched_space = None
    #kernel = C(1.0, (1e-3, 1e3) * RBF(10, (1e-2, 1e2)))
    gp = GaussianProcessRegressor()

    def update_searched_space(self, new_coordinate):
        pos = [0 for i in range(2)]
        for d in range(2):
            pos[d] = int((new_coordinate[d] - self.LBOUNDS[d])/((
             self.LBOUNDS[d] - self.UBOUNDS[d])/5))
        self.searched_space[pos[0]][pos[1]] += 1

    def is_unsearched_space(self):
        #print("min", np.amin(self.searched_space))
        return np.amin(self.searched_space) < np.average(
           self.searched_space)/2

    def add_unsearched_candidate(self):
        self.hyp_pop.clear()
        #print(np.argmin(self.searched_space))
        x, y = divmod(int(np.argmin(self.searched_space)), 5)
        pos = [x, y]
        print(pos)
        for i in range(self.clo_pop.__len__()):
            candidate = []
            mutated_val = 0
            for d in range(2):
                candidate.append(random.uniform(self.LBOUNDS[d] + (
                self.UBOUNDS[d] - self.LBOUNDS[d])/5*pos[d], self.LBOUNDS[d] +
                            (self.UBOUNDS[d] - self.LBOUNDS[d])/5*(pos[d]+1)))
                print(self.LBOUNDS[d] + (
                self.UBOUNDS[d] - self.LBOUNDS[d])/5*pos[d], self.LBOUNDS[d] +
                            (self.UBOUNDS[d] - self.LBOUNDS[d])/5*(pos[d]+1))
            print(candidate)
            if self.fun.number_of_constraints > 0:
                if c <= 0:
                    self.evalcount += 1
                    mutated_val = self.fun(candidate)
                    self.original_coordinates = np.append(
                        self.original_coordinates, [list(
                            candidate.copy())], axis=0)
                    self.original_vals = np.append(self.original_vals,
                                                   mutated_val)
                    self.update_searched_space(candidate)
            else:
                self.evalcount += 1
                mutated_val = self.fun(candidate)

                self.original_coordinates = np.append(
                    self.original_coordinates, [list(
                        candidate.copy())], axis=0)
                self.original_vals = np.append(self.original_vals,
                                               mutated_val)
                self.update_searched_space(candidate)
            self.hyp_pop.append(cell.Cell(candidate.copy(),
                                          mutated_val.copy(), 0))

    def __init__(self, fun, lbounds, ubounds):
        self.fun = fun
        self.LBOUNDS = lbounds
        self.UBOUNDS = ubounds
        self.DIMENSION = len(lbounds)

        self.pop.clear()
        self.clo_pop.clear()
        self.hyp_pop.clear()

        self.original_coordinates = []
        self.original_vals = []
        self.best = None
        self.searched_space = [[0 for i in range(5)] for j in range(5)]
        self.all_best = None
        self.all_best_generation = 0


            #coordinates = np.random.uniform(OptIA.LBOUNDS, OptIA.UBOUNDS,
                                  #OptIA.DIMENSION)
            #coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
            #              np.random.rand(1, self.DIMENSION)
        coordinates = norm.ppf(sobol_seq.i4_sobol_generate(self.DIMENSION,
                                                           OptIA.MAX_POP))*(
            self.UBOUNDS-self.LBOUNDS)/4

        #coordinates = sobol_seq.i4_sobol_generate(self.DIMENSION,
        #                                           int(OptIA.MAX_POP/2))*[5,
         #                                                                -5]
        #coordinates = np.append(coordinates, sobol_seq.i4_sobol_generate(
         #   self.DIMENSION,
          #                                        int(OptIA.MAX_POP/2)) * [
           # -5,5], axis=0)
        #print(self.LBOUNDS, self.UBOUNDS)
        #print(coordinates)
# TODO modify generation phase
        for coordinate in coordinates:
            val = None
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinate)
                if c <= 0:
                    val = self.fun(coordinate)
                    self.evalcount += 1
            else:
                #print(coordinate)
                val = self.fun(coordinate)
                self.evalcount += 1
            #print("Initial eval:", val)
            self.pop.append(cell.Cell(coordinate.copy(), val.copy(), 0))
            self.update_searched_space(coordinate)


    def clone(self, dup):
        self.clo_pop.clear()
        #self.clo_pop = [e for e in copy.deepcopy(self.pop) for _ in range(
        # dup)]
        for i in range(dup):
            c = copy.deepcopy(self.pop)
            for e in c:
                self.clo_pop.append(e)

    def hyper_mutate(self):
        self.hyp_pop.clear()
        np_mutated_coordinates = None
        mutated_coordinates = []
        local_original_vals = []

        for original in self.clo_pop:
            mutated_coordinate = []
            if random.random()< -2.0:
                mutated_coordinate = np.array([original.get_coordinates()[d] +  (self.UBOUNDS[d] -
                                                       self.LBOUNDS[d])/10.0
                                               *random.randint(0, 2)*
                                               random.gauss(0, 1) for d in
                                               range(self.DIMENSION)])
            for d in range(self.DIMENSION):
                val = original.get_coordinates()[d] + (self.UBOUNDS[d] -
                self.LBOUNDS[d])/85.0 * random.randint(1, 3)* random.gauss(
                    0, 1)
                mutated_coordinate = np.append(mutated_coordinate, val)

            while False:
                mutated_coordinate = list(deap.tools.mutGaussian(
                original.get_coordinates().copy(), 0.5, 0.2, 0.5))[0]
                if(all(0 < x for x in (np.array(mutated_coordinate) -
                                 self.LBOUNDS))) and (all(0 < y for y in (
                self.UBOUNDS - np.array(mutated_coordinate)))):
                    break
            while False:
                mutated_coordinate = list(deap.tools.mutPolynomialBounded(
                original.get_coordinates().copy(), eta=0.00000001,
                    low=self.LBOUNDS.tolist(),
                                                     up=self.UBOUNDS.tolist(),
                    indpb=0.5))[0]
                #print("original",original.get_coordinates())
                #print("muta", mutated_coordinate)
                if(all(0 < x for x in (np.array(mutated_coordinate) -
                                 self.LBOUNDS))) and (all(0 < y for y in (
                self.UBOUNDS - np.array(mutated_coordinate)))):
                    break


            if random.random() < 2.7:
                mutated_coordinates += [list(mutated_coordinate.copy())]
            else:
                mutated_coordinates += [list(original.get_coordinates(

                ).copy())]

            np_mutated_coordinates = np.append(np_mutated_coordinates,
                                            mutated_coordinate.copy())

            if self.generation == 0:
                self.best = self.clo_pop[0]
                if len(self.original_coordinates) < 1:
                    self.original_coordinates += [list(original.get_coordinates(
                     ).copy())]
                else:
                    self.original_coordinates = np.append(self.original_coordinates,\
                                                      original.get_array_coordinates(

                                                      ), axis=0)
                self.original_vals = np.append(self.original_vals,
                                          original.get_val())
                local_original_vals = np.append(local_original_vals,
                                               original.get_val())




        #print(original_coordinates)
        #print(original_vals)
        #original_coordinates = np.delete(original_coordinates, 0)
        self.original_coordinates = np.array(self.original_coordinates)
        self.original_coordinates = np.atleast_2d(self.original_coordinates)
        mutated_coordinates = np.atleast_2d(np.array(mutated_coordinates))
        pre_mutated_coordinates = np.delete(np_mutated_coordinates, 0)
        #print(original_coordinates.shape)

        original_coordinates_index = np.unique(self.original_coordinates,
                                              axis=0, return_index=True)[1]
        self.original_coordinates = [self.original_coordinates[
                                         original_coordinates_index] for
                                     original_coordinates_index in sorted(
                original_coordinates_index)]
        #original_vals_index = np.unique(self.original_vals, axis=0,
                                        #return_index=True)[1]
        self.original_vals = [self.original_vals[original_coordinates_index] for
                              original_coordinates_index in sorted(
                                  original_coordinates_index)]

        #print(self.original_coordinates)
        #print("")
        #print(self.original_vals)

        self.gp.fit(self.original_coordinates, self.original_vals)
        vals_pred, sigma = self.gp.predict(mutated_coordinates,
                                           return_std=True)


        #print(vals_pred)
        #print(mutated_coordinates)
        #print(self.clo_pop)

        average = np.average(local_original_vals)

        mutated_val = 0
        for val_pred, mutated_coordinate, original in zip(vals_pred,
                                    mutated_coordinates, self.clo_pop):
            #print("pred", val_pred)
            if ((np.amin(self.best.get_val())> np.amin(val_pred)) or
                self.generation >
                    103) or self.generation < 1: # good
                if self.fun.number_of_constraints > 0:
                    c = self.fun.constraints(mutated_coordinate)
                    if c <= 0:
                        self.evalcount += 1
                        mutated_val = self.fun(mutated_coordinate)
                        self.original_coordinates = np.append(
                            self.original_coordinates, [list(
                                mutated_coordinate.copy())], axis=0)
                        self.original_vals = np.append(self.original_vals,
                                                       mutated_val)
                        #print("real val", mutated_val)
                        self.update_searched_space(mutated_coordinate)
                else:
                    self.evalcount += 1
                    mutated_val = self.fun(mutated_coordinate)
                    self.original_coordinates = np.append(
                        self.original_coordinates, [list(
                            mutated_coordinate.copy())], axis=0)
                    self.original_vals = np.append(self.original_vals,
                                                   mutated_val)
                    self.update_searched_space(mutated_coordinate)
                    #print("real val", mutated_val)

                if np.amin(mutated_val) < np.amin(original.get_val()):
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  mutated_val.copy(), 0))
                else:
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  mutated_val.copy(),
                                                  original.get_age()))
            else:
                if np.amin(val_pred) < np.amin(original.get_val()):
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  val_pred.copy(), 0))
                else:
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  val_pred.copy(),
                                                  original.get_age()))

    def hybrid_age(self):
        for c in self.pop:
            c.add_age()
            if ((OptIA.MAX_AGE < c.get_age()) and (random.random() < 0.5)):
                self.pop.remove(c)
        for c in self.hyp_pop:
            c.add_age()
            if ((OptIA.MAX_AGE < c.get_age()) and (random.random() <  0.5)):
                self.hyp_pop.remove(c)

    def select(self):
        cp = copy.deepcopy(self.hyp_pop)
        for e in cp:
            self.pop.append(e)

        while self.MAX_POP < len(self.pop):
            worst = self.pop[0]
            for c in self.pop:
                #print("pre worst val is ", worst.get_val())
                #print("pre c val is ", c.get_val())
                if np.amin(worst.get_val()) < np.amin(c.get_val()):
                    worst = c
                #print("worst val is ", worst.get_val())
                #print("c val is ", c.get_val())
                #if worst.get_age() < c.get_age():
                 #   worst = c
            self.pop.remove(worst)


        while self.MAX_POP > len(self.pop):
            coordinates = OptIA.LBOUNDS + (OptIA.UBOUNDS - OptIA.LBOUNDS) * \
                          np.random.rand(1, OptIA.DIMENSION)
            val = None
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinates)
                if c <= 0:
                    val = self.fun(coordinates)
            else:
                val = self.fun(coordinates)
            self.pop.append(cell.Cell(coordinates, val, 0))
            self.evalcount += 1

    def opt_ia(self, budget, max_chunk_size):

        # TODO Confirm warnings
        import warnings
        warnings.filterwarnings('ignore')

        t = 0
        best = None
        #print("budget is", budget)
        while budget > 0:
            self.evalcount = 0
            #print("t", t)
            #chunk = int(max([1, min([budget, max_chunk_size])]))
            chunk = self.MAX_POP
            #for c in self.pop:
                #print("before clone", c.get_val())
            self.clone(2)
            if self.is_unsearched_space() and (10 < self.generation) and \
                (10 < self.all_best_generation):
                self.add_unsearched_candidate()
            else:
                self.hyper_mutate()
            self.hybrid_age()
            #for c in self.hyp_pop:
                #print("before select hyp_pop", c.get_val())
            self.select()
            #for c in self.pop:
                #print("after select", c.get_coordinates())
            self.best = self.pop[0]
            for c in self.pop:
                if np.amin(c.get_val()) < np.amin(self.best.get_val()):
                    self.best = c
            #self.best.reset_age()
            print("best is", self.best.get_val())
            print(self.searched_space)
            #print("total pop is", len(self.pop))
            #print("total hyp_pop is", len(self.hyp_pop))
            #print("total clo_pop is", len(self.clo_pop))
            chunk = self.evalcount
            budget -= chunk
            #print("remaining budget ",budget)
            t +=1
            self.generation += 1
            #print("generation", self.generation)
            #print(self.best.get_coordinates())
            #print("test", self.fun([0.83, 0.83]))
            if self.generation == 1:
                self.all_best = self.best
            else:
                if np.amin(self.best.get_val()) > \
                np.amin(self.all_best.get_val()):
                    self.all_best_generation += 1
                else:
                    self.all_best_generation = 0
                    self.all_best = self.best
                #print(np.amin(self.best.get_val()))
                #print(np.amin(self.all_best.get_val()))
                print("all_gn", self.all_best_generation)

        return self.best.get_coordinates()

if __name__ == '__main__':
    # assert len(sys.argv) > 1
    t = 0
    opt_ia = OptIA()
    while t < OptIA.MAX_GENERATION:
        opt_ia.clone(1)
        opt_ia.hyper_mutate()
        opt_ia.hybrid_age()
        opt_ia.select()
        best = opt_ia.pop[0]

        for c in opt_ia.pop:
            if np.amin(c.get_val()) < np.amin(best.get_val()):
                best = c

        #print(best.get_val())
