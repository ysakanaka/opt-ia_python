#!/usr/bin/env python3

import random
import numpy as np
import copy
import cell


class OptIA:
    MAX_GENERATION = 10000
    MAX_POP = 50
    MAX_AGE = 6
    DIMENSION = None
    LBOUNDS = None
    UBOUNDS = None
    fun = None
    evalcount = 0

    GENOTYPE_DUP = True

    pop = []
    clo_pop = []
    hyp_pop = []

    def __init__(self, fun, lbounds, ubounds):
        self.fun = fun
        self.LBOUNDS = lbounds
        self.UBOUNDS = ubounds
        self.DIMENSION = len(lbounds)

        self.pop.clear()
        self.clo_pop.clear()
        self.hyp_pop.clear()
        for i in range(OptIA.MAX_POP):

            #coordinates = np.random.uniform(OptIA.LBOUNDS, OptIA.UBOUNDS,
                                  #OptIA.DIMENSION)
            #coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
            #              np.random.rand(1, self.DIMENSION)

# TODO inplement eval
# TODO modify generation phase
            val = None
            self.evalcount += 1
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinates[0])
                if c <= 0:
                    val = self.fun(coordinates[0])
            else:
                #print(coordinates[0])
                val = self.fun(coordinates[0])
            print("Initial eval:",val)
            self.pop.append(cell.Cell(coordinates[0].copy(), val.copy(), 0))

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
        for original in self.clo_pop:
            mutated_coordinates = original.get_coordinates() + (
                    self.UBOUNDS - self.LBOUNDS)/10000.0 * random.gauss(0, 1)
            # TODO Confirm comparing multiple dimension elements
            if (mutated_coordinates < self.LBOUNDS).all():
                mutated_coordinates = self.LBOUNDS
                print("error")
            elif (mutated_coordinates > self.UBOUNDS).all():
                print("error")
                mutated_coordinates = self.UBOUNDS

            mutated_val = 0

    # TODO implement eval
            print("Coordinates: ",mutated_coordinates)
            self.evalcount += 1
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(mutated_coordinates)
                if c <= 0:
                    mutated_val = self.fun(mutated_coordinates)
            else:
                mutated_val = self.fun(mutated_coordinates)

            #print("mutated val is", mutated_val)
            #print("mutated coordinates is", mutated_coordinates)
            if np.amin(mutated_val) < np.amin(original.get_val()):
                self.hyp_pop.append(cell.Cell(mutated_coordinates.copy(),
                                              mutated_val.copy(), 0))
            else:
                self.hyp_pop.append(cell.Cell(mutated_coordinates.copy(),
                                              mutated_val.copy(),
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
                if np.amin(worst.get_val()) < np.amin(c.get_val()):
                    worst = c
            self.pop.remove(worst)

        while self.MAX_POP > len(self.pop):
            coordinates = OptIA.LBOUNDS + (OptIA.UBOUNDS - OptIA.LBOUNDS) * \
                          np.random.rand(1, OptIA.DIMENSION)
            # TODO inplement eval
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
        t = 0
        best = None
        #print("budget is", budget)
        while budget > 0:
            self.evalcount = 0
            #print("t", t)
            #chunk = int(max([1, min([budget, max_chunk_size])]))
            chunk = self.MAX_POP
            self.clone(2)
            self.hyper_mutate()
            self.hybrid_age()
            self.select()
            best = self.pop[0]
            for c in self.pop:
                if np.amin(c.get_val()) < np.amin(best.get_val()):
                    best = c

            print("best is", best.get_val())
            print("total pop is", len(self.pop))
            print("total hyp_pop is", len(self.hyp_pop))
            print("total clo_pop is", len(self.clo_pop))
            chunk = self.evalcount
            budget -= chunk
            print("remaining budget ",budget)
            t +=1
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
