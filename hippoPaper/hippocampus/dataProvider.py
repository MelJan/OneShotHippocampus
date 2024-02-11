"""  Class for generating datasets with different properties.

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


def flip_states_randomly(pattern, num_states_to_change):
    """  Flips states in the pattern.

    :param pattern: Pattern to change.
    :type pattern: numpy array

    :param num_states_to_change: the number of dimensions which change their state.
    :type num_states_to_change: int

    :return: Pattern with change states.
    :rtype: numpy array
    """
    if num_states_to_change < 0:
        raise Exception("No states flipped, min value is 2 such that one gets active and one gets inactive!")
    indices = numx.argsort(pattern, axis=0)
    border = numx.argmax(numx.sort(pattern, axis=0))
    active_ids = indices[border:]
    inactive_ids = indices[0:border]
    result = numx.copy(pattern)
    active_ids = numx.random.permutation(active_ids)
    inactive_ids = numx.random.permutation(inactive_ids)
    id_max = numx.floor(num_states_to_change / 2.0)
    temp = numx.copy(active_ids[0:id_max])
    active_ids[0:id_max] = inactive_ids[0:id_max]
    inactive_ids[0:id_max] = temp
    result[active_ids] = 1.0
    result[inactive_ids] = 0.0
    return result


def dataset_flip_states_randomly(data, num_states_to_change):
    """  Flips states in the patterns of a dataset..

    :param data: Dataset with pattern to change.
    :type data: numpy array

    :param num_states_to_change: the number of dimensions which change their state.
    :type num_states_to_change: int

    :return: Dataset with patterns with change states.
    :rtype: numpy array
    """
    result = numx.zeros(data.shape)
    for d in range(data.shape[0]):
        result[d] = flip_states_randomly(data[d], num_states_to_change)
    return result


def get_all_sequence_permutations(sequence):
    """ Generates binary random sequence.

    :param sequence: Data sequence.
    :type sequence: numpy array

    :return: All seqence permutations.
    :rtype: numpy array
    """
    data_matrix = None
    row = sequence
    for _ in range(sequence.shape[0]):
        row = numx.roll(row, -1, 0)
        if data_matrix is None:
            data_matrix = row
        else:
            data_matrix = numx.vstack((data_matrix, row))
    return data_matrix


def generate_binary_random_sequence(length_sequence, num_states, num_active_states):
    """ Generates binary random sequence where each pattern has roughly the given number of active units.

    :param length_sequence: Length of the sequence.
    :type length_sequence: int

    :param num_states: Size of the datapoint
    :type num_states: int

    :param num_active_states: Percentage of activity in each pattern.
    :type num_active_states: int

    :return: Array of size  length_subsequence  x num_states
    :rtype: numpy array
    """
    return numx.ones((length_sequence, num_states)) * numx.random.binomial(1, numx.float64(
        num_active_states) / numx.float64(num_states), (length_sequence, num_states))


def generate_binary_random_sequence_probs(length_sequence, probabilities):
    """ Generates binary random sequence follwoing a defined probabilitiy distribution.

    :param length_sequence: Length of the sequence.
    :type length_sequence: int

    :param probabilities: Size of the datapoint.
    :type probabilities: int

    :return: Array of size  length_subsequence  x num_states
    :rtype: numpy array
    """
    pattern = numx.random.binomial(1, probabilities[0], size=(length_sequence, 1))
    for p in range(probabilities.shape[0] - 1):
        pattern = numx.hstack((pattern, numx.random.binomial(1, probabilities[p], size=(length_sequence, 1))))
    return pattern


def generate_binary_random_sequence_fixed_activity(length_sequence, num_states, num_active_states):
    """ Generates binary random sequence where each pattern has a nix number of active units.

    :param length_sequence: Length of the sequence.
    :type length_sequence: int

    :param num_states: Size of the datapoint
    :type num_states: int

    :param num_active_states: Percentage of activity in each pattern.
    :type num_active_states: int

    :return: Array of size  length_subsequence  x num_states
    :rtype: numpy array
    """
    pattern = numx.zeros(num_states)
    pattern[0:num_active_states] = 1.0
    pattern = numx.random.permutation(pattern)
    result = numx.copy(pattern)
    for _ in range(length_sequence - 1):
        pattern = numx.zeros(num_states)
        pattern[0:num_active_states] = 1.0
        pattern = numx.random.permutation(pattern)
        result = numx.vstack((result, numx.copy(pattern)))
    return result


def generate_correlated_binary_random_patterns(length_sequence, num_states, num_active_states, num_states_to_change):
    """ Generates binary random sequence.

    :param length_sequence: Length of the sequence.
    :type length_sequence: int

    :param num_states: Size of the datapoint
    :type num_states: int

    :param num_active_states: Number of activ states
    :type num_active_states: int

    :param num_states_to_change: Number of states to change
    :type num_states_to_change: int

    :return: array of size  length_subsequence  x num_states
    :rtype: numpy array
    """
    org_ids = numx.random.permutation(numx.arange(0, num_states))
    org_active_ids = org_ids[0:num_active_states]
    org_inactive_ids = org_ids[num_active_states:]
    result = None
    for _ in range(length_sequence):
        active_ids = numx.random.permutation(org_active_ids)
        inactive_ids = numx.random.permutation(org_inactive_ids)
        temp = numx.copy(active_ids[0:num_states_to_change / 2])
        active_ids[0:num_states_to_change / 2] = numx.copy(inactive_ids[0:num_states_to_change / 2])
        inactive_ids[0:num_states_to_change / 2] = numx.copy(temp)
        pattern = numx.ones((1, num_states))
        pattern[0, inactive_ids] = 0.0
        if result is None:
            result = pattern
        else:
            result = numx.vstack((result, numx.copy(pattern)))
    return result


def generate_correlated_binary_random_sequence(length_sequence, num_states, num_active_states, num_states_to_change):
    """ Generates binary random sequence.

    :param length_sequence: Length of the sequence.
    :type length_sequence: int

    :param num_states: Size of the datapoint
    :type num_states: int

    :param num_active_states: Number of activ states
    :type num_active_states: int

    :param num_states_to_change: Number of states to change
    :type num_states_to_change: int

    :return: array of size  length_subsequence  x num_states
    :rtype: numpy array
    """
    if num_states_to_change < 0:
        raise Exception("No states flipped, min value is 2 such that one gets active and one gets inactive!")
    ids = numx.random.permutation(numx.arange(0, num_states))
    active_ids = ids[0:num_active_states]
    inactive_ids = ids[num_active_states:]
    pattern = numx.ones((1, num_states))
    pattern[0, inactive_ids] = 0.0
    result = numx.copy(pattern)
    for _ in range(length_sequence - 1):
        active_ids = numx.random.permutation(active_ids)
        inactive_ids = numx.random.permutation(inactive_ids)
        temp = numx.copy(active_ids[0:num_states_to_change / 2])
        active_ids[0:num_states_to_change / 2] = inactive_ids[0:num_states_to_change / 2]
        inactive_ids[0:num_states_to_change / 2] = temp
        pattern = numx.ones((1, num_states))
        pattern[0, inactive_ids] = 0.0
        result = numx.vstack((result, numx.copy(pattern)))
    return result
    """
    change_ids = numx.random.permutation(numx.arange(0,num_states))[0:num_active_units_initially]
    first_pattern = numx.zeros((num_states))
    first_pattern[change_ids] = numx.abs(first_pattern[change_ids] - 1)
    result = numx.copy(first_pattern.reshape(1,first_pattern.shape[0]))
    for _ in range(length_sequences-1):
        first_pattern = numx.copy(result[result.shape[0]-1])
        change_ids = numx.random.permutation(numx.arange(0,num_states))[0:num_states_to_change]
        first_pattern[change_ids] = numx.abs(first_pattern[change_ids] - 1)
        result = numx.vstack((result,first_pattern))
    return result
    """


def generate_binary_blob_chart_sequence(num_charts, num_states, blob_size):
    """ Generates binary sequences where a blob of size 'blob_size' shifts around in the sequence. the indices are
        shuffled 'num_charts' times and the intermediate result is concatenated.

    :Example:
        generate_binary_blob_chart_sequence(2, 3, 5 ):
        --> Original sequence:  11100,01110,00111,10011,11001
            Indices 1:  1->5, 5->2, 2->3, 3->1, 4->4
            Sequence 1: 10101,10110,11010,01011,01101
            Indices 2: ...

    :param num_charts: Number of charts
    :type num_charts: int

    :param num_states: Size of the block/bob or number of active units per pattern.
    :type num_states: int

    :param blob_size: Number of active unit/states per pattern.
    :type blob_size: int

    :return: Array of size  num_charts**num_states  x num_states
    :rtype: numpy array
    """
    data_org = numx.zeros((num_states, num_states))
    data_org[0, 0:blob_size] = numx.ones(blob_size)
    for i in range(1, num_states):
        data_org[i] = numx.roll(data_org[0], i, 0)
    data = None  # data_org
    for _ in range(num_charts):
        index = numx.random.permutation(numx.arange(num_states))
        if data is None:
            data = data_org[:, index]
        else:
            data = numx.vstack((data, data_org[:, index]))
    return data

def Asequence(sequence_length, pattern_dim, activity, stay_active):
    zero_pad = numx.int32(stay_active/activity-stay_active)
    initial_vector = numx.hstack((numx.ones(stay_active), numx.zeros(zero_pad)))
    while len(initial_vector) < sequence_length:
        initial_vector = numx.hstack((initial_vector,numx.ones(stay_active), numx.zeros(zero_pad)))
    initial_vector = initial_vector[0:sequence_length]
    res = numx.copy(initial_vector).reshape(1,sequence_length)
    for i in range(pattern_dim-1):
        res = numx.vstack((res,numx.roll(initial_vector,numx.random.randint(0,sequence_length),0)))
    res = res[0:pattern_dim/8]
    print res.shape
    rnd = numx.random.randint(0,2,(pattern_dim,sequence_length))
    return  numx.vstack((res,rnd))[0:pattern_dim].T


if __name__ == "__main__":
    import pydeep.misc.visualization as vis
    import pydeep.misc.io as io
    from dataEvaluator import *
    numx.random.seed(5624)

    dataset = Asequence(15, 10,0.25, 2)
    print dataset
    print caluclate_average_correlation(dataset), '\t', caluclate_average_temp_correlation(dataset)
    exit()

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
    seq_shift = numx.roll(seq, -1, 0)
    print numx.mean(
        numx.sum(numx.double(numx.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
                 axis=1)), numx.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary uncorrelated random sequence fixed activity: "
    seq = generate_binary_random_sequence_fixed_activity(50, 50, 25)
    vis.imshow_matrix(seq, "Example binary random sequence")
    seq_shift = numx.roll(seq, -1, 0)
    print numx.mean(
        numx.sum(numx.double(numx.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
                 axis=1)), numx.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary temporal correlated random sequence: "
    seq = generate_correlated_binary_random_sequence(50, 50, 25, 2)
    vis.imshow_matrix(seq, "Example binary correlated random sequence")
    seq_shift = numx.roll(seq, -1, 0)
    print numx.mean(
        numx.sum(numx.double(numx.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
                 axis=1)), numx.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary spatial correlated random sequence: "
    seq = numx.random.permutation(generate_correlated_binary_random_sequence(50, 50, 25, 2))
    vis.imshow_matrix(seq, "Example binary correlated random sequence")
    seq_shift = numx.roll(seq, -1, 0)
    print numx.mean(
        numx.sum(numx.double(numx.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
                 axis=1)), numx.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])

    print "Binary blob sequence: "
    seq = generate_binary_blob_chart_sequence(1, 50, 25)
    vis.imshow_matrix(seq, "Example binary blob sequence")
    seq_shift = numx.roll(seq, -1, 0)

    print numx.mean(
        numx.sum(numx.double(numx.logical_xor(seq[0:seq.shape[0] - 1, :], seq_shift[0:seq.shape[0] - 1, :])),
                 axis=1)), numx.mean(seq[0:seq.shape[0] - 1, :] * seq_shift[0:seq.shape[0] - 1, :])
