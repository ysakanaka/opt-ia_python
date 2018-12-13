#!/usr/bin/env python3

import random
import sobol_seq
import numpy as np
import copy
import cell

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class OptIA:
    MAX_GENERATION = 10000
    MAX_POP = 20
    MAX_AGE = 6
    DIMENSION = None
    LBOUNDS = None
    UBOUNDS = None
    fun = None
    evalcount = 0
    generation = 0

    GENOTYPE_DUP = True
    SAIL = True

    pop = []
    clo_pop = []
    hyp_pop = []
    #kernel = C(1.0, (1e-3, 1e3) * RBF(10, (1e-2, 1e2)))
    gp = GaussianProcessRegressor()

    def __init__(self, fun, lbounds, ubounds):
        self.fun = fun
        self.LBOUNDS = lbounds
        self.UBOUNDS = ubounds
        self.DIMENSION = len(lbounds)

        self.pop.clear()
        self.clo_pop.clear()
        self.hyp_pop.clear()



            #coordinates = np.random.uniform(OptIA.LBOUNDS, OptIA.UBOUNDS,
                                  #OptIA.DIMENSION)
            #coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
            #              np.random.rand(1, self.DIMENSION)
        coordinates = sobol_seq.i4_sobol_generate(self.DIMENSION, OptIA.MAX_POP)

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
        original_coordinates = []
        original_vals = []
        for original in self.clo_pop:
            mutated_coordinate = []
            mutated_coordinate = np.array([original.get_coordinates()[d] +  (self.UBOUNDS[d] -
                                                   self.LBOUNDS[d])/100.0 *
                                           random.gauss(0, 1) for d in
                                           range(self.DIMENSION)])
            #for d in range(self.DIMENSION):
            #    val = original.get_coordinates()[d] + (self.UBOUNDS[d] -
            #                                     self.LBOUNDS[d])/100.0 * \
            #    random.gauss(0, 1)
            #    mutated_coordinate = np.append(mutated_coordinate, val)

            # mutated_coordinate = np.delete(mutated_coordinate, 0)
            #print("original", original.get_coordinates())
            #print("mutated", mutated_coordinates)
            if (mutated_coordinate < self.LBOUNDS).all():
                mutated_coordinate = self.LBOUNDS
                print("error")
            elif (mutated_coordinate > self.UBOUNDS).all():
                print("error")
                mutated_coordinate = self.UBOUNDS


            mutated_coordinates += [list(mutated_coordinate.copy())]
            np_mutated_coordinates = np.append(np_mutated_coordinates,
                                            mutated_coordinate.copy())
            #print(mutated_coordinates)
            original_coordinates +=  [list(original.get_coordinates(

            ).copy())]
            #np.append(original_coordinates,
                                   #          np.array(
            #          original.get_coordinates(

                                           #  ).copy()).T)
            original_vals = np.append(original_vals,
                                      original.get_val())

        #print(original_coordinates)
        #print(original_vals)
        #original_coordinates = np.delete(original_coordinates, 0)
        original_coordinates = np.array(original_coordinates)
        original_coordinates = np.atleast_2d(original_coordinates)
        mutated_coordinates = np.atleast_2d(np.array(mutated_coordinates))
        pre_mutated_coordinates = np.delete(np_mutated_coordinates, 0)
        #print(original_coordinates.shape)


        original_vals = original_vals
        #print(original_vals.shape)



        self.gp.fit(original_coordinates, original_vals)
        vals_pred, sigma = self.gp.predict(mutated_coordinates,
                                           return_std=True)


        #print(vals_pred)
        #print(mutated_coordinates)
        #print(self.clo_pop)

        average = np.average(original_vals)

        mutated_val = 0
        for val_pred, mutated_coordinate, original in zip(vals_pred,
                                    mutated_coordinates, self.clo_pop):
            if (average - 0.5 > np.amin(vals_pred)) or self.generation > \
                    100: # good
                if self.fun.number_of_constraints > 0:
                    c = self.fun.constraints(mutated_coordinate)
                    if c <= 0:
                        self.evalcount += 1
                        mutated_val = self.fun(mutated_coordinate)
                else:
                    self.evalcount += 1
                    mutated_val = self.fun(mutated_coordinate)

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
            if ((OptIA.MAX_AGE < c.get_age()) and (random.random() < 1.0 -
                                                  1.0/OptIA.MAX_POP)):
                self.pop.remove(c)
        for c in self.hyp_pop:
            c.add_age()
            if ((OptIA.MAX_AGE < c.get_age()) and (random.random() < 1.0 -
                                                  1.0/OptIA.MAX_POP)):
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
            self.hyper_mutate()
            self.hybrid_age()
            #for c in self.hyp_pop:
                #print("before select hyp_pop", c.get_val())
            self.select()
            #for c in self.pop:
                #print("after select", c.get_val())
            best = self.pop[0]
            for c in self.pop:
                if np.amin(c.get_val()) < np.amin(best.get_val()):
                    best = c
            best.reset_age()
            #print("best is", best.get_val())
            #print("total pop is", len(self.pop))
            #print("total hyp_pop is", len(self.hyp_pop))
            #print("total clo_pop is", len(self.clo_pop))
            chunk = self.evalcount
            budget -= chunk
            #print("remaining budget ",budget)
            t +=1
            self.generation += 1
            #print("generation", self.generation)
            #print(best.get_coordinates())
        return best.get_coordinates()

if __name__ == '__main__':
    # assert len(sys.argv) > 1
    t = 0
    opt_ia = OptIA()
    while t < OptIA.MAX_GENERATION:
        opt_ia.clone(2)
        opt_ia.hyper_mutate()
        opt_ia.hybrid_age()
        opt_ia.select()
        best = opt_ia.pop[0]

        for c in opt_ia.pop:
            if np.amin(c.get_val()) < np.amin(best.get_val()):
                best = c

        #print(best.get_val())
