"""  Class for evalutating datasets with different properties.

    :Version:
        2.0.0

    :Date:
        04.07.2018

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2018 Jan Melchior

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as numx
from dataEvaluator import *
from hippoModel import *
from matplotlib import pyplot as vis

def plot_result(array, base, title, x_lable, y_lable, logscale=False):

    vis.figure(title)

    x = numx.arange(array.shape[1])+1
    y = numx.mean(array, axis=0)
    error = numx.std(array, axis=0)

    vis.fill_between(x, y - error, y + error, alpha=0.3)
    para = numx.polyfit(x, y, 5)
    if logscale:
        vis.loglog(x, y)
        vis.plot(x, para[0] * x ** 5 + para[1] * x ** 4 + para[2] * x ** 3 + para[3] * x ** 2 + para[4] * x + para[5], linestyle='--', linewidth=3,  alpha=0.5)
        legend = ["Reconstruction", "Trend"]
    else:
        vis.plot(x, y)
        vis.plot(x, para[0] * x ** 5 + para[1] * x ** 4 + para[2] * x ** 3 + para[3] * x ** 2 + para[4] * x + para[5], linestyle='--', linewidth=3,  alpha=0.5)
        y = numx.mean(base, axis=0)
        error = numx.std(base, axis=0)
        vis.fill_between(x, y - error, y + error, alpha=0.3)
        vis.plot(x, y, linestyle='-.')
        legend = ["Reconstruction", "Trend", "Baseline"]

    vis.xlabel(x_lable, fontsize=14)
    vis.ylabel(y_lable, fontsize=14)

    vis.ylim(-0.01, 1.01)
    vis.xlim(0,array.shape[1]+1)

    vis.legend(legend)

def get_performance_encoder(models, EC_states, CA3_targets, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    EC_corr_target = numx.zeros((len(models)))
    CA3_corr_target = numx.zeros((len(models)))
    CA3_corr = numx.zeros((len(models)))
    # We have DG
    if isinstance(models[0], HippoECDGCA3EC) or isinstance(models[0], HippoECDGCA3CA1EC):
        DG_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_corr[i] = caluclate_average_correlation(rec_DG_states)
            #print "1",numx.mean(calculate_correlation_two_sequences(rec_DG_states,np.roll(rec_DG_states,1,0)))
            #print "2",numx.mean(calculate_correlation_two_sequences(rec_DG_states,np.roll(rec_DG_states,2,0)))
            rec_CA3_states = models[i].DG_CA3.calculate_output(rec_DG_states)
            result[i] = measure(rec_CA3_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
            CA3_corr[i] = caluclate_average_correlation(rec_CA3_states)
            CA3_corr_target[i] = caluclate_average_correlation(CA3_targets[i])
            result_base[i] = measure(CA3_targets[i], numx.tile(numx.mean(CA3_targets[i],axis=0).reshape(CA3_targets[i].shape[1],1),CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])
        title = "Performance Encoding (EC -> DG -> CA3)"
        print title
        print "Average correlation in EC target:  ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
        print "Average correlation in DG:         ", numx.mean(DG_corr), "(+-",numx.std(DG_corr),")"
    else:
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_CA3_states = models[i].EC_CA3.calculate_output(EC_states[i])
            result[i] = measure(rec_CA3_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
            CA3_corr[i] = caluclate_average_correlation(rec_CA3_states)
            CA3_corr_target[i] = caluclate_average_correlation(CA3_targets[i])
            result_base[i] = measure(CA3_targets[i], numx.tile(numx.mean(CA3_targets[i],axis=0).reshape(CA3_targets[i].shape[1],1),CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])
        title = "Performance Encoding (EC -> CA3)"
        print title
        print "Average correlation in EC target:  ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    print "Average correlation in CA3:        ", numx.mean(CA3_corr), "(+-", numx.std(CA3_corr), ")"
    print "Average correlation in CA3 target: ", numx.mean(CA3_corr_target), "(+-", numx.std(CA3_corr_target), ")"
    plot_result(result, result_base, title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{CA3},\mathbf{x}_t^{CA3})$')

def get_performance_decoder(models, CA3_states, EC_targets, measure):
    result = numx.zeros((len(CA3_states), CA3_states[0].shape[0]))
    result_base = numx.zeros((len(CA3_states), CA3_states[0].shape[0]))
    EC_corr = numx.zeros((len(models)))
    EC_corr_target = numx.zeros((len(models)))
    CA3_corr_target = numx.zeros((len(models)))
    # We have no CA1
    if isinstance(models[0], HippoECDGCA3CA1EC):
        CA1_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            CA3_corr_target[i] = caluclate_average_correlation(CA3_states[i])
            rec_CA1_states = models[i].CA3_CA1.calculate_output(CA3_states[i])
            CA1_corr[i] = caluclate_average_correlation(rec_CA1_states)
            rec_EC_states = models[i].CA1_EC.calculate_input(rec_CA1_states)
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])

            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Decoding (CA3 -> CA1 -> EC)"
        print title
        print "Average correlation in CA3 target: ", numx.mean(CA3_corr_target), "(+-", numx.std(CA3_corr_target), ")"
        print "Average correlation in CA1:        ", numx.mean(CA1_corr), "(+-",numx.std(CA1_corr),")"
    else:
        for i in range(len(models)):
            CA3_corr_target[i] = caluclate_average_correlation(CA3_states[i])
            rec_EC_states = models[i].CA3_EC.calculate_output(CA3_states[i])
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Decoding (CA3 -> EC)"
        print title
        print "Average correlation in CA3 target: ", numx.mean(CA3_corr_target), "(+-", numx.std(CA3_corr_target), ")"
        # We have DG
    print "Average correlation in EC:         ", numx.mean(EC_corr), "(+-", numx.std(EC_corr), ")"
    print "Average correlation in EC target:  ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    plot_result(result,result_base,title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{EC},\mathbf{x}_t^{EC})$')


def get_performance_encoder_decoder(models, EC_states, EC_targets, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    EC_corr_target = numx.zeros((len(models)))
    CA3_corr = numx.zeros((len(models)))
    EC_corr = numx.zeros((len(models)))
    EC_corr_target = numx.zeros((len(models)))
    # We have CA1
    if isinstance(models[0], HippoECDGCA3CA1EC):
        CA1_corr = numx.zeros((len(models)))
        DG_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_corr[i] = caluclate_average_correlation(rec_DG_states)
            rec_CA3_states = models[i].DG_CA3.calculate_output(rec_DG_states)
            CA3_corr[i] = caluclate_average_correlation(rec_CA3_states)
            rec_CA1_states = models[i].CA3_CA1.calculate_output(rec_CA3_states)
            CA1_corr[i] = caluclate_average_correlation(rec_CA1_states)
            rec_EC_states = models[i].CA1_EC.calculate_input(rec_CA1_states)
            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Encoding+Decoding (EC -> DG -> CA3 -> CA1 -> EC)"
        print title
        print "Average correlation in EC t target:   ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
        print "Average correlation in DG t target:   ", numx.mean(DG_corr), "(+-", numx.std(DG_corr), ")"
        print "Average correlation in CA3 t  :       ", numx.mean(CA3_corr), "(+-", numx.std(CA3_corr), ")"
        print "Average correlation in CA1 t+1:       ", numx.mean(CA1_corr), "(+-", numx.std(CA1_corr), ")"
        print "Average correlation in EC t+1:        ", numx.mean(EC_corr), "(+-", numx.std(EC_corr), ")"
        print "Average correlation in EC t+1 target: ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    elif isinstance(models[0], HippoECDGCA3EC):
        DG_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_corr[i] = caluclate_average_correlation(rec_DG_states)
            rec_CA3_states = models[i].DG_CA3.calculate_output(rec_DG_states)
            CA3_corr[i] = caluclate_average_correlation(rec_CA3_states)
            rec_EC_states = models[i].CA3_EC.calculate_output(rec_CA3_states)
            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Encoding+Decoding (EC -> DG -> CA3 -> EC)"
        print title
        print "Average correlation in EC t target:   ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
        print "Average correlation in DG t target:   ", numx.mean(DG_corr), "(+-", numx.std(DG_corr), ")"
        print "Average correlation in CA3 t  :       ", numx.mean(CA3_corr), "(+-", numx.std(CA3_corr), ")"
        print "Average correlation in EC t+1:        ", numx.mean(EC_corr), "(+-", numx.std(EC_corr), ")"
        print "Average correlation in EC t+1 target: ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    elif isinstance(models[0], HippoECCA3EC):
        DG_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_CA3_states = models[i].EC_CA3.calculate_output(EC_states[i])
            CA3_corr[i] = caluclate_average_correlation(rec_CA3_states)
            rec_EC_states = models[i].CA3_EC.calculate_output(rec_CA3_states)
            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Encoding+Decoding (EC -> CA3 -> EC)"
        print title
        print "Average correlation in EC t target:   ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
        print "Average correlation in CA3 t  :       ", numx.mean(CA3_corr), "(+-", numx.std(CA3_corr), ")"
        print "Average correlation in EC t+1:        ", numx.mean(EC_corr), "(+-", numx.std(EC_corr), ")"
        print "Average correlation in EC t+1 target: ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    plot_result(result,result_base, title, 'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{EC},\mathbf{x}_t^{EC})$')

def get_performance_dynamic(models, CA3_states, CA3_targets, measure):
    result = numx.zeros((len(CA3_states), CA3_states[0].shape[0]))
    result_base = numx.zeros((len(CA3_states), CA3_states[0].shape[0]))
    CA3_rec_corr = numx.zeros((len(models)))
    CA3_corr = numx.zeros((len(models)))
    # We have no DG
    for i in range(len(models)):
        CA3_corr[i] = caluclate_average_correlation(CA3_states[i])
        rec_CA3_states = models[i].CA3_CA3.calculate_output(CA3_states[i])
        result[i] = measure(rec_CA3_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
        result_base[i] = measure(CA3_targets[i],
                                 numx.tile(numx.mean(CA3_targets[i], axis=0).reshape(CA3_targets[i].shape[1], 1),
                                           CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])

        CA3_rec_corr[i] = caluclate_average_correlation(rec_CA3_states)
    title = "Performance Dynamics (CA3 t -> CA3 t+1)"
    print title
    print "Average correlation in CA3       : ", numx.mean(CA3_rec_corr), "(+-", numx.std(CA3_rec_corr), ")"
    print "Average correlation in CA3 target: ", numx.mean(CA3_corr), "(+-", numx.std(CA3_corr), ")"
    plot_result(numx.roll(result,+1,1),numx.roll(result_base,+1,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{CA3},\mathbf{x}_t^{CA3})$')

def get_performance_encoder_dynamic(models, EC_states, CA3_targets, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    EC_corr_target = numx.zeros((len(models)))
    CA3_1_corr_target = numx.zeros((len(models)))
    CA3_0_corr = numx.zeros((len(models)))
    CA3_1_corr = numx.zeros((len(models)))
    # We have DG
    if isinstance(models[0], HippoECDGCA3EC) or isinstance(models[0], HippoECDGCA3CA1EC):
        DG_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_corr[i] = caluclate_average_correlation(rec_DG_states)
            rec_CA3_0_states = models[i].DG_CA3.calculate_output(rec_DG_states)
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            result[i] = measure(rec_CA3_1_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
            result_base[i] = measure(CA3_targets[i],
                                     numx.tile(numx.mean(CA3_targets[i], axis=0).reshape(CA3_targets[i].shape[1], 1),
                                               CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])

            CA3_1_corr_target[i] = caluclate_average_correlation(CA3_targets[i])
        title = "Performance Encoding+dynamics (EC -> DG -> CA3 t -> CA3 t+1)"
        print title
        print "Average correlation in EC target:      ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
        print "Average correlation in DG:             ", numx.mean(DG_corr), "(+-",numx.std(DG_corr),")"
    else:
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_CA3_0_states = models[i].EC_CA3.calculate_output(EC_states[i])
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            result[i] = measure(rec_CA3_1_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
            result_base[i] = measure(CA3_targets[i],
                                     numx.tile(numx.mean(CA3_targets[i], axis=0).reshape(CA3_targets[i].shape[1], 1),
                                               CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])

            CA3_1_corr_target[i]= caluclate_average_correlation(CA3_targets[i])
        title = "Performance Encoding+dynamics (EC -> CA3 t -> CA3 t+1)"
        print title
        print "Average correlation in EC target:      ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    print "Average correlation in CA3 t:          ", numx.mean(CA3_0_corr), "(+-", numx.std(CA3_0_corr), ")"
    print "Average correlation in CA3 t+1:        ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
    print "Average correlation in CA3 t+1 target: ", numx.mean(CA3_1_corr_target), "(+-", numx.std(CA3_1_corr_target), ")"
    plot_result(numx.roll(result,+1,1),numx.roll(result_base,+1,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{CA3},\mathbf{x}_t^{CA3})$')

def get_performance_encoder_dynamic_5(models, EC_states, CA3_targets, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    EC_corr_target = numx.zeros((len(models)))
    CA3_1_corr_target = numx.zeros((len(models)))
    CA3_0_corr = numx.zeros((len(models)))
    CA3_1_corr = numx.zeros((len(models)))
    # We have DG
    if isinstance(models[0], HippoECDGCA3EC) or isinstance(models[0], HippoECDGCA3CA1EC):
        DG_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_corr[i] = caluclate_average_correlation(rec_DG_states)
            rec_CA3_0_states = models[i].DG_CA3.calculate_output(rec_DG_states)
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            result[i] = measure(rec_CA3_1_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
            result_base[i] = measure(CA3_targets[i],
                                     numx.tile(numx.mean(CA3_targets[i], axis=0).reshape(CA3_targets[i].shape[1], 1),
                                               CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])

            CA3_1_corr_target[i] = caluclate_average_correlation(CA3_targets[i])
        title = "Performance Encoding+dynamics (EC -> DG -> CA3 t -> ... -> CA3 t+5)"
        print title
        print "Average correlation in EC target:      ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
        print "Average correlation in DG:             ", numx.mean(DG_corr), "(+-",numx.std(DG_corr),")"
    else:
        for i in range(len(models)):
            EC_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_CA3_0_states = models[i].EC_CA3.calculate_output(EC_states[i])
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_1_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            result[i] = measure(rec_CA3_1_states, CA3_targets[i]).reshape(CA3_targets[i].shape[0])
            result_base[i] = measure(CA3_targets[i],
                                     numx.tile(numx.mean(CA3_targets[i], axis=0).reshape(CA3_targets[i].shape[1], 1),
                                               CA3_targets[i].shape[0]).T).reshape(CA3_targets[i].shape[0])

            CA3_1_corr_target[i]= caluclate_average_correlation(CA3_targets[i])
        title = "Performance Encoding+dynamics (EC -> CA3 t -> ... -> CA3 t+5)"
        print title
        print "Average correlation in EC target:      ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    print "Average correlation in CA3 t:          ", numx.mean(CA3_0_corr), "(+-", numx.std(CA3_0_corr), ")"
    print "Average correlation in CA3 t+1:        ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
    print "Average correlation in CA3 t+1 target: ", numx.mean(CA3_1_corr_target), "(+-", numx.std(CA3_1_corr_target), ")"
    plot_result(numx.roll(result,+1,1),numx.roll(result_base,+1,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{CA3},\mathbf{x}_t^{CA3})$')


def get_performance_dynamic_decoder(models, CA3_states, EC_targets, measure):
    result = numx.zeros((len(CA3_states), CA3_states[0].shape[0]))
    result_base = numx.zeros((len(CA3_states), CA3_states[0].shape[0]))
    CA3_0_corr_target = numx.zeros((len(models)))
    CA3_1_corr = numx.zeros((len(models)))
    EC_corr = numx.zeros((len(models)))
    EC_corr_target = numx.zeros((len(models)))
    # We have CA1
    if isinstance(models[0], HippoECDGCA3CA1EC):
        CA1_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            CA3_0_corr_target[i] = caluclate_average_correlation(CA3_states[i])
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(CA3_states[i])
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            rec_CA1_states = models[i].CA3_CA1.calculate_output(rec_CA3_1_states)
            CA1_corr[i] = caluclate_average_correlation(rec_CA1_states)
            rec_EC_states = models[i].CA1_EC.calculate_input(rec_CA1_states)
            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Dynamics+Decoding (CA3 t -> CA3 t+1 -> CA1 -> EC)"
        print title
        print "Average correlation in CA3 target: ", numx.mean(CA3_0_corr_target), "(+-", numx.std(CA3_0_corr_target), ")"
        print "Average correlation in CA3 t+1:        ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
        print "Average correlation in CA1:        ", numx.mean(CA1_corr), "(+-",numx.std(CA1_corr),")"
    else:
        for i in range(len(models)):
            CA3_0_corr_target[i] = caluclate_average_correlation(CA3_states[i])
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(CA3_states[i])
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            rec_EC_states = models[i].CA3_EC.calculate_output(rec_CA3_1_states)
            EC_corr[i] = caluclate_average_correlation(rec_EC_states)
            result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Dynamics+Decoding (CA3 t -> CA3 t+1 -> EC)"
        print title
        print "Average correlation in CA3 target: ", numx.mean(CA3_0_corr_target), "(+-", numx.std(CA3_0_corr_target), ")"
        print "Average correlation in CA3 t+1:        ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
        # We have DG
    print "Average correlation in EC:         ", numx.mean(EC_corr), "(+-", numx.std(EC_corr), ")"
    print "Average correlation in EC target:  ", numx.mean(EC_corr_target), "(+-", numx.std(EC_corr_target), ")"
    plot_result(numx.roll(result,+1,1),numx.roll(result_base,+1,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{EC},\mathbf{x}_t^{EC})$')

def get_performance_encoder_dynamic_decoder(models, EC_states, EC_targets, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    EC_0_corr_target = numx.zeros((len(models)))
    CA3_0_corr = numx.zeros((len(models)))
    CA3_1_corr = numx.zeros((len(models)))
    EC_1_corr = numx.zeros((len(models)))
    EC_1_corr_target = numx.zeros((len(models)))
    # We have CA1
    if isinstance(models[0], HippoECDGCA3CA1EC):
        CA1_1_corr = numx.zeros((len(models)))
        DG_0_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_0_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_0_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_0_corr[i] = caluclate_average_correlation(rec_DG_0_states)
            rec_CA3_0_states = models[i].DG_CA3.calculate_output(rec_DG_0_states)
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            rec_CA1_1_states = models[i].CA3_CA1.calculate_output(rec_CA3_1_states)
            CA1_1_corr[i] = caluclate_average_correlation(rec_CA1_1_states)
            rec_EC_1_states = models[i].CA1_EC.calculate_input(rec_CA1_1_states)
            EC_1_corr[i] = caluclate_average_correlation(rec_EC_1_states)
            result[i] = measure(rec_EC_1_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_1_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Encoding+Dynamics+Decoding (EC -> DG -> CA3 t -> CA3 t+1 -> CA1 -> EC)"
        print title
        print "Average correlation in EC t target:   ", numx.mean(EC_0_corr_target), "(+-", numx.std(EC_0_corr_target), ")"
        print "Average correlation in DG t target:   ", numx.mean(DG_0_corr), "(+-", numx.std(DG_0_corr), ")"
        print "Average correlation in CA3 t  :       ", numx.mean(CA3_0_corr), "(+-", numx.std(CA3_0_corr), ")"
        print "Average correlation in CA3 t+1:       ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
        print "Average correlation in CA1 t+1:       ", numx.mean(CA1_1_corr), "(+-",numx.std(CA1_1_corr),")"
        print "Average correlation in EC t+1:        ", numx.mean(EC_1_corr), "(+-", numx.std(EC_1_corr), ")"
        print "Average correlation in EC t+1 target: ", numx.mean(EC_1_corr_target), "(+-", numx.std(EC_1_corr_target), ")"
    elif isinstance(models[0], HippoECDGCA3EC):
        DG_0_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_0_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_DG_0_states = models[i].EC_DG.calculate_output(EC_states[i])
            DG_0_corr[i] = caluclate_average_correlation(rec_DG_0_states)
            rec_CA3_0_states = models[i].DG_CA3.calculate_output(rec_DG_0_states)
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            rec_EC_1_states = models[i].CA3_EC.calculate_output(rec_CA3_1_states)
            EC_1_corr[i] = caluclate_average_correlation(rec_EC_1_states)
            result[i] = measure(rec_EC_1_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_1_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Encoding+Dynamics+Decodin (EC -> DG -> CA3 t -> CA3 t+1 -> EC)"
        print title
        print "Average correlation in EC t target:   ", numx.mean(EC_0_corr_target), "(+-", numx.std(EC_0_corr_target), ")"
        print "Average correlation in DG t target:   ", numx.mean(DG_0_corr), "(+-", numx.std(DG_0_corr), ")"
        print "Average correlation in CA3 t  :       ", numx.mean(CA3_0_corr), "(+-", numx.std(CA3_0_corr), ")"
        print "Average correlation in CA3 t+1:       ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
        print "Average correlation in EC t+1:        ", numx.mean(EC_1_corr), "(+-", numx.std(EC_1_corr), ")"
        print "Average correlation in EC t+1 target: ", numx.mean(EC_1_corr_target), "(+-", numx.std(EC_1_corr_target), ")"
    elif isinstance(models[0], HippoECCA3EC):
        DG_0_corr = numx.zeros((len(models)))
        for i in range(len(models)):
            EC_0_corr_target[i] = caluclate_average_correlation(EC_states[i])
            rec_CA3_0_states = models[i].EC_CA3.calculate_output(EC_states[i])
            CA3_0_corr[i] = caluclate_average_correlation(rec_CA3_0_states)
            rec_CA3_1_states = models[i].CA3_CA3.calculate_output(rec_CA3_0_states)
            CA3_1_corr[i] = caluclate_average_correlation(rec_CA3_1_states)
            rec_EC_1_states = models[i].CA3_EC.calculate_output(rec_CA3_1_states)
            EC_1_corr[i] = caluclate_average_correlation(rec_EC_1_states)
            result[i] = measure(rec_EC_1_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
            result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
            EC_1_corr_target[i] = caluclate_average_correlation(EC_targets[i])
        title = "Performance Encoding+Dynamics+Decodin (EC -> CA3 t -> CA3 t+1 -> EC)"
        print title
        print "Average correlation in EC t target:   ", numx.mean(EC_0_corr_target), "(+-", numx.std(EC_0_corr_target), ")"
        print "Average correlation in CA3 t  :       ", numx.mean(CA3_0_corr), "(+-", numx.std(CA3_0_corr), ")"
        print "Average correlation in CA3 t+1:       ", numx.mean(CA3_1_corr), "(+-", numx.std(CA3_1_corr), ")"
        print "Average correlation in EC t+1:        ", numx.mean(EC_1_corr), "(+-", numx.std(EC_1_corr), ")"
        print "Average correlation in EC t+1 target: ", numx.mean(EC_1_corr_target), "(+-", numx.std(EC_1_corr_target), ")"
    plot_result(numx.roll(result,+1,1),numx.roll(result_base,+1,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{EC},\mathbf{x}_t^{EC})$')

def get_performance_intrinsic(models, EC_states, EC_targets, steps, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    for i in range(len(models)):
        CA3_states = models[i].encode(EC_states[i])
        for s in range(steps):
            CA3_states = models[i].CA3_CA3.calculate_output(CA3_states)
        rec_EC_states = models[i].decode(CA3_states)
        result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
        result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
        title = "Performance Intrinsic Steps "+str(steps)
        print title
    plot_result(numx.roll(result,steps,1),numx.roll(result_base,steps,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{EC},\mathbf{x}_t^{EC})$')
    #plot_result(result, result_base, title+" (Cue index)", 'Cue index (0 = first stored pattern)', 'Correlation between correct and retrieved pattern')


def get_performance_full_loop(models, EC_states, EC_targets, steps, measure):
    result = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    result_base = numx.zeros((len(EC_states), EC_states[0].shape[0]))
    for i in range(len(models)):
        rec_EC_states = EC_states[i]
        for s in range(steps):
            CA3_states = models[i].encode(rec_EC_states)
            CA3_states = models[i].CA3_CA3.calculate_output(CA3_states)
            rec_EC_states = models[i].decode(CA3_states)
        result[i] = measure(rec_EC_states, EC_targets[i]).reshape(EC_targets[i].shape[0])
        result_base[i] = measure(EC_targets[i],
                                     numx.tile(numx.mean(EC_targets[i], axis=0).reshape(EC_targets[i].shape[1], 1),
                                               EC_targets[i].shape[0]).T).reshape(EC_targets[i].shape[0])
        title = "Performance Full Loop Steps "+str(steps)
        print title
    plot_result(numx.roll(result,steps,1),numx.roll(result_base,steps,1),title,'Pattern index t ('+str(result.shape[1])+' = latest pattern)', r'$Corr(\mathbf{\dot{x}}_t^{EC},\mathbf{x}_t^{EC})$')
    #plot_result(result, result_base, title+" (Cue index)", 'Cue index (0 = first stored pattern)', 'Correlation between correct and retrieved pattern')

