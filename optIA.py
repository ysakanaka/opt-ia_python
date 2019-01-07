#!/usr/bin/env python3

import logging
import random
import sobol_seq
import numpy as np
import copy
import cell

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
import deap.tools

logger = logging.getLogger("optIA")


class OptIA:
    MAX_GENERATION = 1000000000
    MAX_POP = 20
    MAX_AGE = 10
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
    SURROGATE_ASSIST = False
    RESET_AGE = False
    SEARCHSPACE_ASSIST = False
    SOBOL_SEQ_GENERATION = True

    pop = []
    clo_pop = []
    hyp_pop = []

    original_coordinates = []
    original_vals = []
    searched_space = None
    gp = GaussianProcessRegressor()

    def update_searched_space(self, new_coordinate):
        pos = [0 for i in range(2)]
        for d in range(2):
            pos[d] = int((new_coordinate[d] - self.LBOUNDS[d])/((
             self.UBOUNDS[d] - self.LBOUNDS[d])/5))
            if 4 < pos[d]:
                pos[d] = 4
            elif pos[d] < -4:
                pos[d] = -4  # TODO should be modify
        try:
            self.searched_space[pos[0]][pos[1]] += 1

        except IndexError:
            print(new_coordinate)
            print(pos)
            exit()

    def is_unsearched_space(self):
        return np.amin(self.searched_space) < np.average(
           self.searched_space)/1.5

    def add_unsearched_candidate(self):  # TODO new candidates are not added
        # into original_coodrinates
        self.hyp_pop.clear()
        x, y = divmod(int(np.argmin(self.searched_space)), 5)
        pos = [x, y]
        for i in range(self.clo_pop.__len__()):
            candidate = []
            mutated_val = 0
            for d in range(2):
                candidate.append(random.uniform(self.LBOUNDS[d] + (
                    self.UBOUNDS[d] - self.LBOUNDS[d])/5*pos[d],
                    self.LBOUNDS[d] + (self.UBOUNDS[d] - self.LBOUNDS[
                        d])/5*(pos[d]+1)))

            if self.SURROGATE_ASSIST:
                self.gp.fit(self.original_coordinates, self.original_vals)
                vals_pred, deviations = self.gp.predict([candidate],
                                                        return_std=True)
                if deviations[0] < 2 and np.amin(self.best.get_val()) < \
                        np.amin(vals_pred[0]):
                    self.update_searched_space(candidate)
                    self.hyp_pop.append(cell.Cell(candidate.copy(),
                                                  vals_pred[0].copy(), 0))
                    continue


            if self.SURROGATE_ASSIST:
                self.gp.fit(self.original_coordinates, self.original_vals)
                vals_pred, deviations = self.gp.predict([candidate],
                                                        return_std=True)
                if deviations[0] < 2 and np.amin(self.best.get_val()) < \
                        np.amin(vals_pred[0]):
                    self.update_searched_space(candidate)
                    self.hyp_pop.append(cell.Cell(candidate.copy(),
                                                  vals_pred[0].copy(), 0))
                    continue

            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(candidate)
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

    def __init__(self, fun, lbounds, ubounds, ra=False, ssa=False,
                 sua=False, sobol=True):
        self.fun = fun
        self.LBOUNDS = lbounds
        self.UBOUNDS = ubounds
        self.DIMENSION = len(lbounds)

        self.RESET_AGE = ra
        self.SEARCHSPACE_ASSIST = ssa
        self.SURROGATE_ASSIST = sua
        self.SOBOL_SEQ_GENERATION = sobol
        self.pop.clear()
        self.clo_pop.clear()
        self.hyp_pop.clear()

        self.original_coordinates = []
        self.original_vals = []
        self.best = None
        self.searched_space = [[0 for i in range(5)] for j in range(5)]
        self.all_best = None
        self.all_best_generation = 0

        # TODO Confirm the parameters for sobol_seq
        if self.SOBOL_SEQ_GENERATION:
            coordinates = norm.ppf(sobol_seq.i4_sobol_generate(self.DIMENSION,
                                                           OptIA.MAX_POP))*(
             self.UBOUNDS-self.LBOUNDS)/4
        else:
            coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
                          np.random.rand(OptIA.MAX_POP, self.DIMENSION)

        # TODO modify generation phase
        for coordinate in coordinates:
            val = None
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinate)
                if c <= 0:
                    val = self.fun(coordinate)
                    self.evalcount += 1
            else:
                val = self.fun(coordinate)
                self.evalcount += 1
            self.pop.append(cell.Cell(coordinate.copy(), val.copy(), 0))
            self.update_searched_space(coordinate)

    def clone(self, dup):
        self.clo_pop.clear()
        for i in range(dup):
            c = copy.deepcopy(self.pop)
            for e in c:
                self.clo_pop.append(e)

    def hyper_mutate_master(self):
        self.hyp_pop.clear()
        mutated_coordinates = []

        for original in self.clo_pop:
            mutated_coordinate = []
            if random.random() < -2.0:
                mutated_coordinate = np.array([original.get_coordinates()[d]
                                               + (self.UBOUNDS[d]
                                                  - self.LBOUNDS[d]) / 10.0 *
                                               random.randint(0, 2) *
                                               random.gauss(0, 1) for d in
                                               range(self.DIMENSION)])

            for d in range(self.DIMENSION):
                val = original.get_coordinates()[d] + (self.UBOUNDS[d] -
                                                       self.LBOUNDS[d]) / 85.0 \
                      * random.randint(2, 3) * random.gauss(0, 1)
                mutated_coordinate = np.append(mutated_coordinate, val)

            while False:
                mutated_coordinate = list(deap.tools.mutGaussian(
                    original.get_coordinates().copy(), 0.5, 0.2, 0.5))[0]
                if (all(0 < x for x in (np.array(mutated_coordinate) -
                                        self.LBOUNDS))) and (all(0 < y for y
                                                                 in (
                                                                         self.UBOUNDS - np.array(
                                                                     mutated_coordinate)))):
                    break
            while True:
                mutated_coordinate = list(deap.tools.mutPolynomialBounded(
                    original.get_coordinates().copy(), eta=0.00000001,
                    low=self.LBOUNDS.tolist(), up=self.UBOUNDS.tolist(),
                    indpb=0.5))[0]
                logger.critical('mutated values %s', mutated_coordinates)
                if (all(0 < x for x in (np.array(mutated_coordinate) -
                                        self.LBOUNDS))) and (all(0 < y for y in
                            (self.UBOUNDS - np.array(mutated_coordinate)))):
                    break

            mutated_coordinates += [list(mutated_coordinate.copy())]

            if self.generation == 0:
                self.best = self.clo_pop[0]
                self.original_coordinates += [list(original.get_coordinates(
                     ).copy())]
                self.original_vals = np.append(self.original_vals,
                                               original.get_val())

        self.original_coordinates = np.array(self.original_coordinates)
        self.original_coordinates = np.atleast_2d(self.original_coordinates)
        mutated_coordinates = np.atleast_2d(np.array(mutated_coordinates))

        original_coordinates_index = np.unique(self.original_coordinates,
                                               axis=0, return_index=True)[1]
        self.original_coordinates = [self.original_coordinates[
                                         original_coordinates_index] for
                                     original_coordinates_index in sorted(
                original_coordinates_index)]
        self.original_vals = [self.original_vals[original_coordinates_index]
                              for
                              original_coordinates_index in sorted(
                original_coordinates_index)]
        mutated_val = 0

        vals_pred = []
        deviations = []

        if self.SURROGATE_ASSIST:
            self.gp.fit(self.original_coordinates, self.original_vals)
            vals_pred, deviations = self.gp.predict(mutated_coordinates,
                                                    return_std=True)
        else:
            vals_pred = mutated_coordinates
            deviations = mutated_coordinates

    # TODO implement eval
        logger.critical('mutated_coordinate len %s',
                        mutated_coordinates.__len__())
        logger.critical('original len  %s', self.clo_pop.__len__())
        logger.critical('val_pred len %s', vals_pred.__len__())
        logger.critical('deviation len %s', deviations.__len__())
        for mutated_coordinate, original, val_pred, deviation, in zip(
                mutated_coordinates, self.clo_pop, vals_pred, deviations):
            #print("Coordinates: ",mutated_coordinates)
            logger.critical('mutated_coordinate %s', mutated_coordinate)
            logger.critical('original %s', original)
            logger.critical('val_pred %s', val_pred)
            logger.critical('deviation %s', deviation)


            self.evalcount += 1
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(mutated_coordinate)
                if c <= 0:
                    mutated_val = self.fun(mutated_coordinate)
                    self.original_coordinates = np.append(
                        self.original_coordinates, [list(
                            mutated_coordinate.copy())], axis=0)
                    self.original_vals = np.append(self.original_vals,
                                                   mutated_val)
                    self.update_searched_space(mutated_coordinate)
            else:
                mutated_val = self.fun(mutated_coordinate)
                self.original_coordinates = np.append(
                    self.original_coordinates, [list(
                        mutated_coordinate.copy())], axis=0)
                self.original_vals = np.append(self.original_vals,
                                               mutated_val)

            if np.amin(mutated_val) < np.amin(original.get_val()):
                self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                              mutated_val.copy(), 0))
            else:
                self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                              mutated_val.copy(),
                                              original.get_age()))

    def hyper_mutate(self):
        self.hyp_pop.clear()
        mutated_coordinates = []

        for original in self.clo_pop:
            mutated_coordinate = []
            if random.random() < -2.0:
                mutated_coordinate = np.array([original.get_coordinates()[d]
                                               + (self.UBOUNDS[d]
                                                  - self.LBOUNDS[d])/10.0 *
                                               random.randint(0, 2) *
                                               random.gauss(0, 1) for d in
                                               range(self.DIMENSION)])

            for d in range(self.DIMENSION):
                val = original.get_coordinates()[d] + (self.UBOUNDS[d] -
                                                    self.LBOUNDS[d])/85.0 \
                        * random.randint(2, 3) * random.gauss(0, 1)
                mutated_coordinate = np.append(mutated_coordinate, val)

            while False:
                mutated_coordinate = list(deap.tools.mutGaussian(
                    original.get_coordinates().copy(), 0.5, 0.2, 0.5))[0]
                if(all(0 < x for x in (np.array(mutated_coordinate) -
                        self.LBOUNDS))) and (all(0 < y for y
                        in (self.UBOUNDS - np.array(mutated_coordinate)))):
                    break
            while True:
                mutated_coordinate = list(deap.tools.mutPolynomialBounded(
                    original.get_coordinates().copy(), eta=0.00000001,
                    low=self.LBOUNDS.tolist(), up=self.UBOUNDS.tolist(),
                    indpb=0.5))[0]
                if(all(0 < x for x in (np.array(mutated_coordinate) -
                            self.LBOUNDS))) and (all(0 < y for y in
                            (self.UBOUNDS - np.array(mutated_coordinate)))):
                    break

            mutated_coordinates += [list(mutated_coordinate.copy())]

            if self.generation == 0:
                self.best = self.clo_pop[0]
                self.original_coordinates += [list(original.get_coordinates(
                     ).copy())]
                self.original_vals = np.append(self.original_vals,
                                               original.get_val())

        self.original_coordinates = np.array(self.original_coordinates)
        self.original_coordinates = np.atleast_2d(self.original_coordinates)
        mutated_coordinates = np.atleast_2d(np.array(mutated_coordinates))

        original_coordinates_index = np.unique(self.original_coordinates,
                                               axis=0, return_index=True)[1]
        self.original_coordinates = [self.original_coordinates[
                                         original_coordinates_index] for
                                     original_coordinates_index in sorted(
                original_coordinates_index)]
        self.original_vals = [self.original_vals[original_coordinates_index] for
                              original_coordinates_index in sorted(
                                  original_coordinates_index)]

        vals_pred = []
        deviations = []

        if self.SURROGATE_ASSIST:
            self.gp.fit(self.original_coordinates, self.original_vals)
            vals_pred, deviations = self.gp.predict(mutated_coordinates,
                                               return_std=True)
        else:
            vals_pred = mutated_coordinates
            deviations = mutated_coordinates

        mutated_val = 0
        for mutated_coordinate, original, val_pred, deviation, in zip(
                mutated_coordinates, self.clo_pop, vals_pred, deviations):
            if self.SURROGATE_ASSIST:
                if ((np.amin(self.best.get_val()) > np.amin(val_pred)) or (
                        3 < deviation) or self.generation > 50000):
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
            else:
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
                if np.amin(mutated_val) < np.amin(original.get_val()):
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  mutated_val.copy(), 0))
                else:
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
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
            coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
                          np.random.rand(1, self.DIMENSION)
            val = None
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinates)
                if c <= 0:
                    val = self.fun(coordinates)
            else:
                val = self.fun(coordinates[0])
            self.pop.append(cell.Cell(coordinates[0], val, 0))
            self.evalcount += 1

    def opt_ia(self, budget):  # TODO Chunk system
        logging.basicConfig()
        logging.getLogger("optIA").setLevel(level=logging.CRITICAL)
        # TODO Confirm warnings
        import warnings
        warnings.filterwarnings('ignore')
        logger.debug('budget is %s', budget)
        while budget > 0:
            self.evalcount = 0
            self.clone(2)
            if self.SEARCHSPACE_ASSIST:  # TODO Condition?
                if self.is_unsearched_space() and (10 < self.generation) and \
                            (100 < self.all_best_generation):
                    self.add_unsearched_candidate()
                else:
                    self.hyper_mutate()
            else:
                self.hyper_mutate()
            self.hybrid_age()
            self.select()
            self.best = self.pop[0]
            for c in self.pop:
                if np.amin(c.get_val()) < np.amin(self.best.get_val()):
                    self.best = c
            if OptIA.RESET_AGE:
                self.best.reset_age()

            chunk = self.evalcount
            budget -= chunk

            logger.debug(self.searched_space)
            logger.debug(self.pop.__len__())
            logger.debug(self.hyp_pop.__len__())
            logger.debug(self.clo_pop.__len__())
            logger.debug(budget)

            self.generation += 1

            logger.debug(self.generation)
            logger.debug(self.best.get_coordinates())

            if self.generation == 1:
                self.all_best = self.best
            else:
                if np.amin(self.best.get_val()) >= \
                        np.amin(self.all_best.get_val()):
                    self.all_best_generation += 1
                else:
                    self.all_best_generation = 0
                    self.all_best = self.best

                logger.debug(np.amin(self.best.get_val()))
                logger.debug(np.amin(self.all_best.get_val()))
                logger.debug('all_best %s', self.all_best_generation)
                logger.debug('budget is %s', budget)
                logger.debug('generation is %s', self.generation)
                logger.debug(self.all_best_generation)
                logger.debug('budget is %s', budget)

        return self.best.get_coordinates()