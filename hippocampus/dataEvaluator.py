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
import numpy as np
#import pydeep.misc.visualization as vis
#from dataProvider import get_all_sequence_permutations
from hippoModel import HippoECCA3EC,HippoECDGCA3EC,HippoECDGCA3CA1EC

def collision_check(sequence):
    """ Checks if all elements in a sequence are unique.

    :param sequence: Sequence to check.
    :type sequence: numpy array num_sequences x num_states

    :return: True if a collision is present False otherwise.
    :rtype: bool
    """
    for i in range(0, sequence.shape[0]):
        for u in range(i + 1, sequence.shape[0]):
            if np.sum(np.abs(sequence[i] - sequence[u])) <= 0:
                return True
    return False


def caluclate_average_correlation(data):
    """ Calculates the average correlations coefficient.

    :param data: Dataset.
    :type data: numpy array

    :return: average correlation coefficient.
    :rtype: float
    """
    N = data.shape[1]
    return (np.sum(np.corrcoef(data))-N)/(N*N-N)


def calculate_correlation_two_sequences(seq1, seq2):
    """ Calculates the correlations coefficients of two sequences, i.e. average correlation of corr(x_0,y_0),
        corr(x_1,y_1) ... corr(x_T,y_T).

    :param seq1: Sequence 1.
    :type seq1: numpy array

    :param seq2: Sequence 2.
    :type seq2: numpy array

    :return: Average temporal correlations.
    :rtype: float
    """
    mean1 = np.mean(seq1, axis=1).reshape(seq1.shape[0], 1)
    mean2 = np.mean(seq2, axis=1).reshape(seq1.shape[0], 1)
    std1 = np.std(seq1, axis=1).reshape(seq1.shape[0], 1)
    std2 = np.std(seq2, axis=1).reshape(seq1.shape[0], 1)
    nom = np.mean((seq1 - mean1) * (seq2 - mean2), axis=1).reshape(seq1.shape[0], 1)
    #print np.sum(mean1), np.sum(mean2),np.sum(std1), np.sum(std2),np.sum(nom),np.sum(std1 * std2),np.sum(nom / (std1 * std2))
    return nom / (std1 * std2)


def calculate_max_cross_correlation(seq1, seq2):
    """ Calculates the auto-correlations of two sequences, i.e. average correlation of max(corr(x_0,y_t)),
        max(corr(x_1,y_t)) ... max(corr(x_T,y_t)).

    :param seq1: Sequence 1.
    :type seq1: numpy array

    :param seq2: Sequence 2.
    :type seq2: numpy array

    :return: Average auto correlations.
    :rtype: float
    """
    res = calculate_correlation_two_sequences(seq1, seq2)
    print res.shape
    for i in range(seq1.shape[0]):
        temp_res = calculate_correlation_two_sequences(seq1, np.roll(seq2, -i, 0))
        for u in range(seq1.shape[0]):
            if res[u] < temp_res[u]:
                res[u] = temp_res[u]
    return res

def calculate_auto_correlation_two_sequences(seq1, seq2):
    """ Calculates the auto-correlations of two sequences, i.e. average correlation of max(corr(x_0,y_t)),
        max(corr(x_1,y_t)) ... max(corr(x_T,y_t)).

    :param seq1: Sequence 1.
    :type seq1: numpy array

    :param seq2: Sequence 2.
    :type seq2: numpy array

    :return: Average auto correlations.
    :rtype: float
    """
    res = np.mean(calculate_correlation_two_sequences(seq1, seq2))
    for i in range(seq1.shape[0]):
        temp_res = np.mean(calculate_correlation_two_sequences(seq1, np.roll(seq2, -i, 0)))
        if res < temp_res:
            res = temp_res
    return res

def caluclate_average_temp_correlation(data):
    """ Calculates the average temporal correlations coefficient i.e. average correlation of x_t and x_t+1.

    :param data: Dataset.
    :type data: numpy array

    :return: Average temporal correlations.
    :rtype: float
    """
    return np.mean(calculate_correlation_two_sequences(data[0:data.shape[0] - 1], data[1:data.shape[0]]))

def get_total_variance(data, actitvity):
    print np.sum(np.var(data,axis = 1)),data.shape[1]*(actitvity-actitvity**2)

def get_data_correlation(data, actitvity):
    print np.mean(np.corrcoef(data)),np.std(np.corrcoef(data))

def get_maximum_paarwise_correlation(data):
    '''
    max_paarwise = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        current_max = -9999999.0
        for j in range(data.shape[0]):
            corr = np.mean((data[i]-np.mean(data[i]))*(data[j]-np.mean(data[j])))/(np.sqrt(np.mean(data[i]-np.mean(data[i])**2))*np.sqrt(np.mean(data[j]-np.mean(data[j])**2)))
            if i != j and corr > current_max:
                current_max = corr
        max_paarwise[i] = current_max

    print max_paarwise

    max_paarwise = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        current_max = -9999999.0
        for j in range(0,data.shape[0]):
            corr = np.corrcoef(data[i],data[j])[0,1]
            if i != j and corr > current_max:
                current_max = corr
        max_paarwise[i] = current_max

    print max_paarwise
    '''
    return np.max(np.corrcoef(data)-np.eye(data.shape[0],data.shape[0]),axis = 0)


def show_results(hippo, data):
    enc = calculate_correlation_two_sequences(hippo.encode(data), hippo.CA3_CA3._training_sequence)
    dec = calculate_correlation_two_sequences(hippo.decode(hippo.CA3_CA3._training_sequence), data)
    enc_dec = calculate_correlation_two_sequences(hippo.encode_decode(data), data)
    enc_pre_dec = calculate_correlation_two_sequences(hippo.encode_next_decode(np.roll(data, 1, 0)), data)

    print "Correlation of"
    print "Enc\tDec\tEnc-Dec\tEnc-Next-Dec"
    print "Last 10 sample"
    N = 10
    print np.mean(enc),
    print np.mean(dec[data.shape[0] - N:]),
    print np.mean(enc_dec[data.shape[0] - N:]),
    print np.mean(enc_pre_dec[data.shape[0] - N:])

    print "Last 20 sample"
    N = 20
    print np.mean(enc[data.shape[0] - N:]),
    print np.mean(dec[data.shape[0] - N:]),
    print np.mean(enc_dec[data.shape[0] - N:]),
    print np.mean(enc_pre_dec[data.shape[0] - N:])

    print "Last 50 sample"
    N = 50
    print np.mean(enc[data.shape[0] - N:]),
    print np.mean(dec[data.shape[0] - N:]),
    print np.mean(enc_dec[data.shape[0] - N:]),
    print np.mean(enc_pre_dec[data.shape[0] - N:])

    print "ALL"
    print np.mean(enc),
    print np.mean(dec),
    print np.mean(enc_dec),
    print np.mean(enc_pre_dec)
    full_loop = hippo.full_loop_encode_next_decode(data, True)
    intr_loop = hippo.full_intrinsic_loop_encode_next_decode(data, True)
    target = get_all_sequence_permutations(data)

    print "Correlation of Full Loop\tIntr. loop"
    print np.mean(calculate_correlation_two_sequences(full_loop, target)),
    print np.mean(calculate_correlation_two_sequences(intr_loop, target))

    rec_intr_loop = []
    rec_rev_intr_loop = []
    rec_full_loop = []
    tar_intr_loop = []
    tar_rev_intr_loop = []
    tar_full_loop = []
    for i in range(hippo.CA3_capacity):
        rec_intr_loop.append(hippo.full_intrinsic_loop_encode_next_decode(data[i], True))
        rec_rev_intr_loop.append(hippo.full_intrinsic_loop_encode_next_decode(data[i], False))
        rec_full_loop.append(hippo.full_loop_encode_next_decode(data[i], True))
        tar_intr_loop.append(np.roll(data, -i - 1, 0))
        tar_rev_intr_loop.append(np.roll(data, -i, 0)[::-1])
        tar_full_loop.append(np.roll(data, -i - 1, 0))

    result_corr = np.zeros((hippo.CA3_capacity, 24))
    print "Single values!"
    print "Pattern/Cue index\tCorr\tAuto corr."
    for i in range(hippo.CA3_capacity):
        length = 1
        result_corr[i, 0] = np.mean(
            calculate_correlation_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length]))
        result_corr[i, 1] = np.mean(
            calculate_correlation_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length]))
        result_corr[i, 2] = np.mean(
            calculate_correlation_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length]))

        length = 5
        result_corr[i, 3] = np.mean(
            calculate_correlation_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length]))
        result_corr[i, 4] = np.mean(
            calculate_correlation_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length]))
        result_corr[i, 5] = np.mean(
            calculate_correlation_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length]))

        length = 10
        result_corr[i, 6] = np.mean(
            calculate_correlation_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length]))
        result_corr[i, 7] = np.mean(
            calculate_correlation_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length]))
        result_corr[i, 8] = np.mean(
            calculate_correlation_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length]))

        length = 20
        result_corr[i, 9] = np.mean(
            calculate_correlation_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length]))
        result_corr[i, 10] = np.mean(
            calculate_correlation_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length]))
        result_corr[i, 11] = np.mean(
            calculate_correlation_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length]))

        length = 40
        result_corr[i, 12] = np.mean(
            calculate_correlation_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length]))
        result_corr[i, 13] = np.mean(
            calculate_correlation_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length]))
        result_corr[i, 14] = np.mean(
            calculate_correlation_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length]))

        length = np.int32(hippo.CA3_capacity / 2)
        result_corr[i, 15] = np.mean(
            calculate_correlation_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length]))
        result_corr[i, 16] = np.mean(
            calculate_correlation_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length]))
        result_corr[i, 17] = np.mean(
            calculate_correlation_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length]))

        result_corr[i, 18] = np.mean(calculate_correlation_two_sequences(rec_intr_loop[i], tar_intr_loop[i]))
        result_corr[i, 19] = np.mean(calculate_correlation_two_sequences(rec_rev_intr_loop[i], tar_rev_intr_loop[i]))
        result_corr[i, 20] = np.mean(calculate_correlation_two_sequences(rec_full_loop[i], tar_full_loop[i]))

        result_corr[i, 21] = np.mean(calculate_auto_correlation_two_sequences(rec_intr_loop[i], tar_intr_loop[i]))
        result_corr[i, 22] = np.mean(
            calculate_auto_correlation_two_sequences(rec_rev_intr_loop[i], tar_rev_intr_loop[i]))
        result_corr[i, 23] = np.mean(calculate_auto_correlation_two_sequences(rec_full_loop[i], tar_full_loop[i]))

        print i, "FORI", result_corr[i, 0], "\t", \
            result_corr[i, 3], "\t", \
            result_corr[i, 6], "\t", \
            result_corr[i, 9], "\t", \
            result_corr[i, 12], "\t", \
            result_corr[i, 15], "\t", \
            result_corr[i, 18], "\t", \
            result_corr[i, 21]
        print i, "REVI", result_corr[i, 0 + 1], "\t", \
            result_corr[i, 3 + 1], "\t", \
            result_corr[i, 6 + 1], "\t", \
            result_corr[i, 9 + 1], "\t", \
            result_corr[i, 12 + 1], "\t", \
            result_corr[i, 15 + 1], "\t", \
            result_corr[i, 18 + 1], "\t", \
            result_corr[i, 21 + 1]
        print i, "FULL", result_corr[i, 0 + 2], "\t", \
            result_corr[i, 3 + 2], "\t", \
            result_corr[i, 6 + 2], "\t", \
            result_corr[i, 9 + 2], "\t", \
            result_corr[i, 12 + 2], "\t", \
            result_corr[i, 15 + 2], "\t", \
            result_corr[i, 18 + 2], "\t", \
            result_corr[i, 21 + 2]
    legendPos = 0
    if isinstance(hippo,HippoECCA3EC):
        vis.figure("Performance Encoding (EC -> CA3)")
    else:
        vis.figure("Performance Encoding (EC -> DG -> CA3)")
    vis.plot(enc)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    if isinstance(hippo,HippoECDGCA3CA1EC):
        vis.figure("Performance Decoding (CA3 -> CA1 -> EC)")
    else:
        vis.figure("Performance Decoding (CA3 -> EC)")
    vis.plot(dec)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    if isinstance(hippo,HippoECCA3EC):
        vis.figure("Performance Encoding + Decoding (EC -> CA3 -> EC)")
    else:
        if isinstance(hippo, HippoECDGCA3EC):
            vis.figure("Performance Encoding + Decoding (EC -> DG -> CA3 -> EC)")
        else:
            vis.figure("Performance Encoding + Decoding (EC -> DG -> CA3 -> CA1 -> EC)")
    vis.plot(enc_dec)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)

    if isinstance(hippo,HippoECCA3EC):
        vis.figure("Performance Encoding + Predict + Decoding (EC -> CA3 -> CA3 -> EC)")
    else:
        if isinstance(hippo, HippoECDGCA3EC):
            vis.figure("Performance Encoding + Predict + Decoding (EC -> DG -> CA3 -> CA3 -> EC)")
        else:
            vis.figure("Performance Encoding + Predict + Decoding (EC -> DG -> CA3 -> CA3 -> CA1 -> EC)")
    vis.plot(enc_pre_dec)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    vis.figure("Performance Intrinsic - 1 Step")
    vis.plot(result_corr[:, 0:3])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    vis.figure("Performance Intrinsic - 5 Step")
    vis.plot(result_corr[:, 3:6])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    vis.figure("Performance Intrinsic - 10 Step")
    vis.plot(result_corr[:, 6:9])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    vis.figure("Performance Intrinsic - half sequence length")
    vis.plot(result_corr[:, 15:18])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    vis.figure("Performance Intrinsic - full sequence length")
    vis.plot(result_corr[:, 18:21])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Correlation')

    vis.figure("Performance auto correlation Intrinsic - full sequence length")
    vis.plot(result_corr[:, 21:24])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Average Auto Correlation')

    vis.show()


def compare_two_sequences(seq1,seq2):
    return calculate_correlation_two_sequences(seq1, seq2)


def show_results2(hippo, data):

    # Evaluate encoder performance
    rec_CA3_1 = hippo.encode(data)
    rec_CA3_2 = hippo.CA3_CA3.calculate_output(rec_CA3_1)
    rec_CA3_2 = hippo.CA3_CA3.calculate_output(hippo.encode(data))

    enc = compare_two_sequences(hippo.encode(data), hippo.CA3_CA3._training_sequence)
    dec = compare_two_sequences(hippo.decode(hippo.CA3_CA3._training_sequence), data)
    enc_dec = compare_two_sequences(hippo.encode_decode(data), data)
    enc_pre_dec = compare_two_sequences(hippo.encode_next_decode(np.roll(data, 1, 0)), data)

    print "Correlation of"
    print "Enc\tDec\tEnc-Dec\tEnc-Next-Dec"
    print "Last 10 sample"
    N = 10
    print np.mean(enc),
    print np.mean(dec[data.shape[0] - N:]),
    print np.mean(enc_dec[data.shape[0] - N:]),
    print np.mean(enc_pre_dec[data.shape[0] - N:])

    print "Last 20 sample"
    N = 20
    print np.mean(enc[data.shape[0] - N:]),
    print np.mean(dec[data.shape[0] - N:]),
    print np.mean(enc_dec[data.shape[0] - N:]),
    print np.mean(enc_pre_dec[data.shape[0] - N:])

    print "Last 50 sample"
    N = 50
    print np.mean(enc[data.shape[0] - N:]),
    print np.mean(dec[data.shape[0] - N:]),
    print np.mean(enc_dec[data.shape[0] - N:]),
    print np.mean(enc_pre_dec[data.shape[0] - N:])

    print "ALL"
    print np.mean(enc),
    print np.mean(dec),
    print np.mean(enc_dec),
    print np.mean(enc_pre_dec)
    full_loop = hippo.full_loop_encode_next_decode(data, True)
    intr_loop = hippo.full_intrinsic_loop_encode_next_decode(data, True)
    target = get_all_sequence_permutations(data)

    print "Correlation of Full Loop\tIntr. loop"
    print np.mean(compare_two_sequences(full_loop, target)),
    print np.mean(compare_two_sequences(intr_loop, target))

    rec_intr_loop = []
    rec_rev_intr_loop = []
    rec_full_loop = []
    tar_intr_loop = []
    tar_rev_intr_loop = []
    tar_full_loop = []
    for i in range(hippo.CA3_capacity):
        rec_intr_loop.append(hippo.full_intrinsic_loop_encode_next_decode(data[i], True))
        rec_rev_intr_loop.append(hippo.full_intrinsic_loop_encode_next_decode(data[i], False))
        rec_full_loop.append(hippo.full_loop_encode_next_decode(data[i], True))
        tar_intr_loop.append(np.roll(data, -i - 1, 0))
        tar_rev_intr_loop.append(np.roll(data, -i, 0)[::-1])
        tar_full_loop.append(np.roll(data, -i - 1, 0))

    result_corr = np.zeros((hippo.CA3_capacity, 24))
    print "Single values!"
    print "Pattern/Cue index\tCorr\tAuto corr."
    for i in range(hippo.CA3_capacity):
        length = 1
        result_corr[i, 0] = compare_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length])[-1]
        result_corr[i, 1] = compare_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length])[-1]
        result_corr[i, 2] = compare_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length])[-1]

        length = 2
        result_corr[i, 3] = compare_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length])[-1]
        result_corr[i, 4] = compare_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length])[-1]
        result_corr[i, 5] = compare_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length])[-1]

        length = 5
        result_corr[i, 6] = compare_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length])[-1]
        result_corr[i, 7] = compare_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length])[-1]
        result_corr[i, 8] = compare_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length])[-1]

        length = 10
        result_corr[i, 9] = compare_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length])[-1]
        result_corr[i, 10] = compare_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length])[-1]
        result_corr[i, 11] = compare_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length])[-1]

        length = 20
        result_corr[i, 12] = compare_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length])[-1]
        result_corr[i, 13] = compare_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length])[-1]
        result_corr[i, 14] = compare_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length])[-1]

        length = np.int32(hippo.CA3_capacity / 2)
        result_corr[i, 15] = compare_two_sequences(rec_intr_loop[i][0:length], tar_intr_loop[i][0:length])[-1]
        result_corr[i, 16] = compare_two_sequences(rec_rev_intr_loop[i][0:length], tar_rev_intr_loop[i][0:length])[-1]
        result_corr[i, 17] = compare_two_sequences(rec_full_loop[i][0:length], tar_full_loop[i][0:length])[-1]

        result_corr[i, 18] = compare_two_sequences(rec_intr_loop[i], tar_intr_loop[i])[-1]
        result_corr[i, 19] = compare_two_sequences(rec_rev_intr_loop[i], tar_rev_intr_loop[i])[-1]
        result_corr[i, 20] = compare_two_sequences(rec_full_loop[i], tar_full_loop[i])[-1]

        result_corr[i, 21] = compare_two_sequences(rec_intr_loop[i], tar_intr_loop[i])[-1]
        result_corr[i, 22] = compare_two_sequences(rec_rev_intr_loop[i], tar_rev_intr_loop[i])[-1]
        result_corr[i, 23] = compare_two_sequences(rec_full_loop[i], tar_full_loop[i])[-1]

        print i, "FORI", result_corr[i, 0], "\t", \
            result_corr[i, 3], "\t", \
            result_corr[i, 6], "\t", \
            result_corr[i, 9], "\t", \
            result_corr[i, 12], "\t", \
            result_corr[i, 15], "\t", \
            result_corr[i, 18], "\t", \
            result_corr[i, 21]
        print i, "REVI", result_corr[i, 0 + 1], "\t", \
            result_corr[i, 3 + 1], "\t", \
            result_corr[i, 6 + 1], "\t", \
            result_corr[i, 9 + 1], "\t", \
            result_corr[i, 12 + 1], "\t", \
            result_corr[i, 15 + 1], "\t", \
            result_corr[i, 18 + 1], "\t", \
            result_corr[i, 21 + 1]
        print i, "FULL", result_corr[i, 0 + 2], "\t", \
            result_corr[i, 3 + 2], "\t", \
            result_corr[i, 6 + 2], "\t", \
            result_corr[i, 9 + 2], "\t", \
            result_corr[i, 12 + 2], "\t", \
            result_corr[i, 15 + 2], "\t", \
            result_corr[i, 18 + 2], "\t", \
            result_corr[i, 21 + 2]
    legendPos = 0
    if isinstance(hippo,HippoECCA3EC):
        vis.figure("Performance Encoding (EC -> CA3)")
    else:
        vis.figure("Performance Encoding (EC -> DG -> CA3)")
    vis.plot(enc)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    #vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    if isinstance(hippo,HippoECDGCA3CA1EC):
        vis.figure("Performance Decoding (CA3 -> CA1 -> EC)")
    else:
        vis.figure("Performance Decoding (CA3 -> EC)")
    vis.plot(dec)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    #vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    if isinstance(hippo,HippoECCA3EC):
        vis.figure("Performance Encoding + Decoding (EC -> CA3 -> EC)")
    else:
        if isinstance(hippo, HippoECDGCA3EC):
            vis.figure("Performance Encoding + Decoding (EC -> DG -> CA3 -> EC)")
        else:
            vis.figure("Performance Encoding + Decoding (EC -> DG -> CA3 -> CA1 -> EC)")
    vis.plot(enc_dec)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.ylabel('Correlation between true and predicted pattern')
    #vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)

    if isinstance(hippo,HippoECCA3EC):
        vis.figure("Performance Encoding + Predict + Decoding (EC -> CA3 -> CA3 -> EC)")
    else:
        if isinstance(hippo, HippoECDGCA3EC):
            vis.figure("Performance Encoding + Predict + Decoding (EC -> DG -> CA3 -> CA3 -> EC)")
        else:
            vis.figure("Performance Encoding + Predict + Decoding (EC -> DG -> CA3 -> CA3 -> CA1 -> EC)")
    vis.plot(enc_pre_dec)
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    #vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.figure("Performance Intrinsic - 1 Step")
    vis.plot(result_corr[:, 0:3])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.figure("Performance Intrinsic - 5 Step")
    vis.plot(result_corr[:, 3:6])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.figure("Performance Intrinsic - 10 Step")
    vis.plot(result_corr[:, 6:9])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.figure("Performance Intrinsic - half sequence length")
    vis.plot(result_corr[:, 15:18])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.figure("Performance Intrinsic - full sequence length")
    vis.plot(result_corr[:, 18:21])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.figure("Performance auto correlation Intrinsic - full sequence length")
    vis.plot(result_corr[:, 21:24])
    vis.ylim(0.0, 1.0)
    vis.xlim(hippo.CA3_capacity, 0.0)
    vis.legend(["Intrinsic Predict", "Intrinsic Remember", "Full loop Predict"],loc=legendPos)
    vis.xlabel('Pattern index (0 = first stored pattern)')
    vis.ylabel('Correlation between true and predicted pattern')

    vis.show()


if __name__ == "__main__":
    import pydeep.misc.visualization as vis
    import pydeep.misc.io as io
    from dataProvider import *

    np.random.seed(211)

    dataset = generate_binary_random_sequence_fixed_activity(2000, 1100, 385)
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)

    dataset = generate_correlated_binary_random_sequence(2000, 1100, 385, 1)
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)

    dataset = generate_correlated_binary_random_sequence(2000, 1100, 385, 2)
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)

    dataset = generate_correlated_binary_random_patterns(2000, 1100, 385, 302)
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)

    dataset = io.load_object('DATA_GRID_OUTPUT', True, False)[0, 0:2000, :]
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)

    dataset = io.load_object('DATA_GRID_INPUT', True, False)[0, 0:2000, :]
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)

    exit()
    print "Binary uncorrelated random sequence: "
    seq = generate_binary_random_sequence(50, 50, 25)
    vis.imshow_matrix(seq, "Example binary random sequence")
    seq_shift = np.roll(seq, -1, 0)
    print np.mean(
        np.sum(np.double(np.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
               axis=1)), np.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary uncorrelated random sequence fixed activity: "
    seq = generate_binary_random_sequence_fixed_activity(50, 50, 25)
    vis.imshow_matrix(seq, "Example binary random sequence")
    seq_shift = np.roll(seq, -1, 0)
    print np.mean(
        np.sum(np.double(np.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
               axis=1)), np.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary temporal correlated random sequence: "
    seq = generate_correlated_binary_random_sequence(50, 50, 25, 2)
    vis.imshow_matrix(seq, "Example binary correlated random sequence")
    seq_shift = np.roll(seq, -1, 0)
    print np.mean(
        np.sum(np.double(np.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
               axis=1)), np.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary spatial correlated random sequence: "
    seq = np.random.permutation(generate_correlated_binary_random_sequence(50, 50, 25, 2))
    vis.imshow_matrix(seq, "Example binary correlated random sequence")
    seq_shift = np.roll(seq, -1, 0)
    print np.mean(
        np.sum(np.double(np.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
               axis=1)), np.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary blob sequence: "
    seq = generate_binary_blob_chart_sequence(1, 50, 25)
    vis.imshow_matrix(seq, "Example binary blob sequence")
    seq_shift = np.roll(seq, -1, 0)

    print np.mean(
        np.sum(np.double(np.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
               axis=1)), np.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])




