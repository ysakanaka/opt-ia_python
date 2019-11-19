#!/usr/bin/env python3

import logging
import random
import sobol_seq
import numpy as np
import copy
import cell
import plot

from collections import OrderedDict
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
import deap.tools

logger = logging.getLogger("optIA")


class OptIA:
    MUT_GAUSSIAN = 0
    MUT_POLYNOMIAL_BOUNDED = 1
    GRADIENT_DESCENT = 2

    def convert_dict_to_array(self, d, explored_coordinates, explored_vals,
                              coordinates):
        for key in d.keys():
            if type(d[key]) != np.float64:
                self.convert_dict_to_array(d[key], explored_coordinates,
                                           explored_vals, np.append(coordinates, key))
            else:
                self.explored_coordinates = \
                    np.append(self.explored_coordinates, [np.append(
                        coordinates, key)], axis=0)
                self.explored_vals = np.append(self.explored_vals, (d[key]))

    def pickup_values(self, explored_coordinates, explored_vals):
        clength = len(explored_coordinates)
        picked_coordinates = np.zeros((1000, len(explored_coordinates[0])))
        picked_vals = np.zeros(1000)
        if clength <= 1000:
            return
        else:
            j = 0
            indexs = np.random.choice(
                clength, 1000, replace=False)
            for i in indexs:
                picked_coordinates[j] = explored_coordinates[i]
                picked_vals[j] = explored_vals[i]

                j += 1

        self.explored_coordinates = picked_coordinates
        self.explored_vals = picked_vals
        return

    def add_points_into_dict(self, d, new_coordinate, new_val):
        if new_coordinate[0] in d:
            if len(new_coordinate) == 1:
                return
            else:
                self.add_points_into_dict(
                    d[new_coordinate[0]], new_coordinate[1:], new_val)
        else:
            d[new_coordinate[0]] = OrderedDict()
            if len(new_coordinate) == 1:
                d[new_coordinate[0]] = new_val
                return
            else:
                self.add_points_into_dict(d[new_coordinate[0]],
                                          new_coordinate[1:], new_val)

    def store_explored_points(self, new_coordinate, new_val):
        new_coordinate = \
            list(map(
                lambda x: round(x, self.ROUNDIN_NUM_DIGITS), new_coordinate))
        self.add_points_into_dict(
            self.explored_points, new_coordinate, new_val)

        pos = [0, 0]
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
                q, mod = divmod(self.generation, 1)
                if self.generation < 20:
                    #print("Correct update of the GP")
                    self.gp.fit(self.explored_coordinates, self.explored_vals)
                elif mod == 0:
                    #print("Correct gp update")
                    self.gp.fit(self.explored_coordinates, self.explored_vals)
                vals_pred, deviations = self.gp.predict([candidate],
                                                        return_std=True)
                if deviations[0] < 3 and np.amin(self.best.val) < \
                        np.amin(vals_pred[0]):
                    self.store_explored_points(candidate, vals_pred[0].copy())
                    self.hyp_pop.append(cell.Cell(candidate.copy(),
                                                  vals_pred[0].copy(), 0))
                    continue

            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(candidate)
                if c <= 0:
                    mutated_val = self.my_fun(candidate)
                    self.store_explored_points(candidate.copy(),
                                               mutated_val.copy())
            else:
                mutated_val = self.my_fun(candidate)
                self.store_explored_points(candidate.copy(),
                                           mutated_val.copy())
            self.hyp_pop.append(cell.Cell(candidate.copy(),
                                          mutated_val.copy(), 0))

    def my_fun(self, x):
        y = self.fun(x)
        self.evalcount += 1
        if self.fun.final_target_hit and not self.target_hit_first:
            self.all_best = cell.Cell(x, y, 0)
            logger.debug("best sol in my fun %s", y)
            self.target_hit_first = True
        return y

    def __init__(self, fun, lbounds, ubounds, ra=False,
                 ssa=False,
                 sua=False, sobol=True, gd=False):

        self.MAX_GENERATION = 1000000000
        self.MAX_POP = 30
        self.MAX_AGE = 10
        self.evalcount = 0
        self.generation = 0
        self.ROUNDIN_NUM_DIGITS = 12
        self.GENOTYPE_DUP = True
        self.pop = []
        self.clo_pop = []
        self.hyp_pop = []
        self.gp = GaussianProcessRegressor(kernel=1**2 * Matern(
            length_scale=2, nu=1.5))
        self.fun = fun
        self.target_hit_first = False
        self.LBOUNDS = lbounds
        self.UBOUNDS = ubounds
        self.DIMENSION = len(lbounds)
        self.RESET_AGE = ra
        self.SEARCHSPACE_ASSIST = ssa
        self.SURROGATE_ASSIST = sua
        self.SOBOL_SEQ_GENERATION = sobol
        self.GRADIENT_DESCENT = gd
        self.MUTATION = OptIA.MUT_POLYNOMIAL_BOUNDED
        self.pop.clear()
        self.clo_pop.clear()
        self.hyp_pop.clear()
        self.explored_coordinates = []
        self.explored_vals = []
        self.explored_points = OrderedDict()
        self.best = None
        self.searched_space = [[0 for _i in range(5)] for _j in range(5)]
        self.all_best = None
        self.all_best_generation = 0
        self.stocked_value = 0
        self.predicted_coordinates = []
        self.predicted_vals = []

        # TODO Confirm the parameters for sobol_seq
        if self.SOBOL_SEQ_GENERATION:
            coordinates = sobol_seq.i4_sobol_generate(
                self.DIMENSION, self.MAX_POP)*(
                    self.UBOUNDS-self.LBOUNDS)+self.LBOUNDS
        else:
            coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
                          np.random.rand(self.MAX_POP, self.DIMENSION)

        # TODO modify generation phase
        for coordinate in coordinates:
            val = None
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinate)
                if c <= 0:
                    val = self.my_fun(coordinate)
            else:
                val = self.my_fun(coordinate)
            self.pop.append(cell.Cell(coordinate.copy(), val.copy(), 0))
            self.store_explored_points(coordinate.copy(), val.copy())

        self.best = copy.deepcopy(self.pop[0])

    def calculate_gradient(self, x):
        h = 1e-4
        x = np.array(x)
        gradient = np.zeros_like(x)
        for i in range(x.size):
            store_x = x[:]

            # f(x+h)
            x[i] += h
            vals_pred, deviation = self.gp.predict([x], return_std=True)
            if (1 + self.generation / 5000) > deviation[0]:
                f_x_plus_h = vals_pred[0]
            else:
                f_x_plus_h = self.my_fun(x)

            x = store_x[:]

            # f(x-h)
            x[i] -= h
            vals_pred, deviation = self.gp.predict([x], return_std=True)
            if (1 + self.generation / 5000) > deviation[0]:
                f_x_minus_h = vals_pred[0]
            else:
                f_x_minus_h = self.my_fun(x)
            gradient[i] = (f_x_plus_h - f_x_minus_h)/(2*h)

        return gradient

    def gradient_descent(self, x):
        max_iter = 100
        learning_rate = 0.1
        for i in range(max_iter):
            via = (learning_rate * self.calculate_gradient(x))
            x -= via
            for j in range(self.DIMENSION):
                if x[j] < self.LBOUNDS[j] or x[j] > self.UBOUNDS[j]:
                    x += via
                    break
            else:
                continue
            break

        return x

    def clone(self, dup):
        self.clo_pop.clear()
        for i in range(dup):
            c = copy.deepcopy(self.pop)
            for e in c:
                self.clo_pop.append(e)

    def hyper_mutate(self):
        self.hyp_pop.clear()
        mutated_coordinates = []
        etalist = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001,
                   0.0000001, 0.00000001, 0.000000001,
                   0.00000000001]
        etalistcount = 0
        for original in self.clo_pop:
            mutated_coordinate = []

            if random.random() < 0.5 or not self.GRADIENT_DESCENT:
                if self.MUTATION == OptIA.MUT_GAUSSIAN:
                    while True:
                        mutated_coordinate = list(deap.tools.mutGaussian(
                            original.coordinates.copy(), 0.5, 0.2, 0.5))[0]
                        if(all(0 < x for x in (np.array(mutated_coordinate) -
                                               self.LBOUNDS))) \
                                and (all(0 < y for y in
                                         (self.UBOUNDS -
                                          np.array(mutated_coordinate)))):
                            break
                elif self.MUTATION == OptIA.MUT_POLYNOMIAL_BOUNDED:
                    while True:
                        mutated_coordinate \
                            = list(deap.tools.mutPolynomialBounded(
                                original.coordinates.copy(),
                                eta=random.random()/etalist[etalistcount],
                                low=self.LBOUNDS.tolist(),
                                up=self.UBOUNDS.tolist(), indpb=0.8))[0]
                        etalistcount = (etalistcount+1) % 10
                        #print(self.UBOUNDS.tolist())
                        #print(self.LBOUNDS.tolist())
                        if(all(0 < x for x in
                               (np.array(mutated_coordinate) - self.LBOUNDS))) \
                                and (all(0 < y for y
                                         in (self.UBOUNDS -
                                             np.array(mutated_coordinate)))):
                            break

            else:
                if self.GRADIENT_DESCENT:
                    mutated_coordinate = \
                        self.gradient_descent(original.coordinates)

            mutated_coordinates += [list(mutated_coordinate.copy())]

            if self.generation == 0:
                self.best = self.clo_pop[0]

        mutated_coordinates = np.atleast_2d(np.array(mutated_coordinates))

        if self.SURROGATE_ASSIST:
            q, mod = divmod(self.generation, 10)
            stock_value = self.explored_coordinates.__len__()
            if self.generation < 20:
                self.explored_coordinates = np.empty((0, self.DIMENSION),
                                                     np.float64)
                self.explored_vals = np.empty((0, self.DIMENSION), np.float64)
                self.convert_dict_to_array(self.explored_points,
                                           self.explored_coordinates,
                                           self.explored_vals, np.array([]))
                self.pickup_values(self.explored_coordinates, self.explored_vals)
                self.gp.fit(self.explored_coordinates,
                            self.explored_vals.reshape(-1, 1))
                self.stocked_value = stock_value
            elif mod == 0:
                self.explored_coordinates = np.empty((0, self.DIMENSION),
                                                     np.float64)
                self.explored_vals = np.empty((0, self.DIMENSION),
                                              np.float64)
                self.convert_dict_to_array(self.explored_points,
                                           self.explored_coordinates,
                                           self.explored_vals, np.array([]))
                self.pickup_values(self.explored_coordinates,
                                   self.explored_vals)
                logger.debug("picked coordinates %s", self.explored_coordinates)
                self.gp.fit(self.explored_coordinates,
                            self.explored_vals.reshape(-1, 1))
                self.stocked_value = stock_value
            vals_pred, deviations = self.gp.predict(mutated_coordinates,
                                                    return_std=True)
            self.predicted_coordinates = mutated_coordinates
            self.predicted_vals = vals_pred
        else:
            vals_pred = mutated_coordinates
            deviations = mutated_coordinates
        logger.debug("best sol at the middle %s", self.best.val)

        mutated_val = 0
        for mutated_coordinate, original, val_pred, deviation, in zip(
                mutated_coordinates, self.clo_pop, vals_pred, deviations):
            if self.SURROGATE_ASSIST:
                #logger.debug("predicted %s %s", val_pred, deviation)
                #logger.debug("actual %s", self.fun(mutated_coordinate))
                if ((np.amin(self.best.val) > np.amin(val_pred)) or (
                        1/(1+20*original.age) < deviation) or
                        self.generation > 50000):
                    #print(deviation)

                    if self.fun.number_of_constraints > 0:
                        c = self.fun.constraints(mutated_coordinate)
                        if c <= 0:
                            mutated_val = self.my_fun(mutated_coordinate)
                            self.store_explored_points(mutated_coordinate,
                                                       mutated_val)
                    else:
                        mutated_val = self.my_fun(mutated_coordinate)
                        self.store_explored_points(mutated_coordinate,
                                                   mutated_val)
                    if np.amin(mutated_val) < np.amin(original.val):
                        self.hyp_pop.append(
                            cell.Cell(mutated_coordinate.copy(),
                                      mutated_val.copy(), 0))
                    else:
                        self.hyp_pop.append(
                            cell.Cell(mutated_coordinate.copy(),
                                      mutated_val.copy(), original.age))
                else:
                    if np.amin(val_pred) < np.amin(original.val):
                        self.hyp_pop.append(
                            cell.Cell(mutated_coordinate.copy(),
                                      val_pred.copy(), 0))
                    else:
                        self.hyp_pop.append(
                            cell.Cell(mutated_coordinate.copy(),
                                      val_pred.copy(), original.age))
            else:
                if self.fun.number_of_constraints > 0:
                    c = self.fun.constraints(mutated_coordinate)
                    if c <= 0:
                        mutated_val = self.my_fun(mutated_coordinate)
                        self.store_explored_points(mutated_coordinate,
                                                   mutated_val)
                else:
                    mutated_val = self.my_fun(mutated_coordinate)
                    self.store_explored_points(mutated_coordinate, mutated_val)
                if np.amin(mutated_val) < np.amin(original.val):
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  mutated_val.copy(), 0))
                else:
                    self.hyp_pop.append(cell.Cell(mutated_coordinate.copy(),
                                                  mutated_val.copy(),
                                                  original.age))

    def hybrid_age(self):
        for c in self.pop:
            c.add_age()
            if ((self.MAX_AGE < c.age) and (random.random() < 1.0 -
                                                  1.0/self.MAX_POP)):
                self.pop.remove(c)
        for c in self.hyp_pop:
            c.add_age()
            if ((self.MAX_AGE < c.age) and (random.random() < 1.0 -
                                                  1.0/self.MAX_POP)):
                self.hyp_pop.remove(c)

    def select(self):
        cp = copy.deepcopy(self.hyp_pop)
        for e in cp:
            self.pop.append(e)

        while self.MAX_POP < len(self.pop):
            worst = self.pop[0]
            for c in self.pop:
                if np.amin(worst.val) < np.amin(c.val):
                    worst = c
            self.pop.remove(worst)

        while self.MAX_POP > len(self.pop):
            coordinates = self.LBOUNDS + (self.UBOUNDS - self.LBOUNDS) * \
                          np.random.rand(1, self.DIMENSION)
            val = None
            if self.fun.number_of_constraints > 0:
                c = self.fun.constraints(coordinates)
                if c <= 0:
                    val = self.my_fun(coordinates)
            else:
                val = self.my_fun(coordinates[0])
            self.pop.append(cell.Cell(np.array(coordinates[0]), val, 0))

    def opt_ia(self, budget):  # TODO Chunk system
        logging.basicConfig()
        logging.getLogger("optIA").setLevel(level=logging.CRITICAL)
        xx, yy = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5))
        latticePoints = np.c_[xx.ravel(), yy.ravel()]
        # TODO Confirm warnings
        import warnings
        warnings.filterwarnings('ignore')
        logger.debug('budget is %s', budget)
        myplot = plot.Plot()
        while budget > 0 and not self.fun.final_target_hit:
            logger.debug('Generation at loop start is %s', self.generation)
            logger.debug('Generation at loop start is %s', self.generation)
            if self.generation % 100 == 0 and False:
                myplot.plot(\
                    self.explored_coordinates,
                                  self.explored_vals,
                        "Detected points")
            if False and self.generation % 10 == 0 and \
                    self.predicted_coordinates.__len__() > 1 and \
                    self.predicted_vals.__len__() > 1:
                myplot.plot(self.predicted_coordinates, self.predicted_vals,
                            "Predicted points")
                predicted_points, _ = self.gp.predict(latticePoints,
                                                        return_std=True)
                myplot.plot(latticePoints, predicted_points,
                            "Predicted points on lattice")

            self.evalcount = 0
            self.clone(2)
            if self.SEARCHSPACE_ASSIST:  # TODO Condition?
                if self.is_unsearched_space() and (10 < self.generation) and \
                            (10 < self.all_best_generation):
                    self.add_unsearched_candidate()
                else:
                    self.hyper_mutate()
            else:
                self.hyper_mutate()
            logger.debug("best sol after hypermut %s", self.best.val)
            self.hybrid_age()
            logger.debug("best sol after hybridage %s", self.best.val)
            self.select()
            logger.debug("best sol after select %s", self.best.val)
            for c in self.pop:
                logger.debug("each individuals %s", c.val)
                if np.amin(c.val) < np.amin(self.best.val):
                    self.best = c
                    logger.debug("inserted")
            if self.RESET_AGE:
                self.best.reset_age()
            logger.debug("best sol after all %s", self.best.val)
            chunk = self.evalcount
            budget -= chunk

            logger.debug(self.searched_space)
            logger.debug('stock values length %s',
                         self.explored_coordinates.__len__())
            logger.debug(self.pop.__len__())
            logger.debug(self.hyp_pop.__len__())
            logger.debug(self.clo_pop.__len__())

            self.generation += 1

            logger.debug('generation is %s',self.generation)
            logger.debug('budget is %s', budget)
            logger.debug(self.best.coordinates)

            if self.generation == 1:
                self.all_best = copy.deepcopy(self.best)
            else:
                if np.amin(self.best.val) >= \
                        np.amin(self.all_best.val):
                    self.all_best_generation += 1
                else:
                    self.all_best_generation = 0
                    self.all_best = copy.deepcopy(self.best)

                logger.debug(np.amin(self.best.val))
                logger.debug(np.amin(self.all_best.val))
                logger.debug('all_best %s', self.all_best_generation)
                logger.debug('budget is %s', budget)
                logger.debug('generation is %s', self.generation)
                logger.debug(self.all_best_generation)
            logger.debug('Generation at end of loop is %s', self.generation)




        return self.all_best.coordinates
