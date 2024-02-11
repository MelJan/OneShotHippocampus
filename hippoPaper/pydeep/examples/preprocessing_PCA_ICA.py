''' Example for the preprocessing functions on natural images.

    :Version:
        1.0

    :Date:
        06.06.2016

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2016 Jan Melchior

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

'''
import numpy

import pydeep.misc.visualization as STATISTICS
import pydeep.misc.visualization as VISUALIZATION

numpy.random.seed(42)

# You'll get the data from https://zenodo.org/record/167823
#data = scipy.io.loadmat('../../../data/NaturalImage.mat')['rawImages'].T[0:10000,:]
import pydeep.misc.io as IO
data = numpy.random.permutation(IO.load_object('../../../data/nat_traj.arr',compressed=False))[0:10000,:]
data = preprocessing.remove_rows_means(data)

d = 8
# PCA
pca = preprocessing.PCA(data.shape[1])
pca.train(data)
filters = VISUALIZATION.tile_matrix_rows(pca.projection_matrix, d,d,d,d, border_size = 1,normalized = False)
VISUALIZATION.imshow_matrix(filters, 'Projection Matrix PCA')

# ZCA
zca = preprocessing.ZCA(data.shape[1])
zca.train(data)
filters = VISUALIZATION.tile_matrix_rows(zca.projection_matrix, d,d,d,d, border_size = 1,normalized = False)
VISUALIZATION.imshow_matrix(filters, 'Projection Matrix ZCA')

# ICA
data = zca.project(data)
ica = preprocessing.ICA(data.shape[1])
ica.train(data,1000,0.1,status = True)
filters = VISUALIZATION.tile_matrix_rows(ica.projection_matrix, d,d,d,d, border_size = 1,normalized = False)
VISUALIZATION.imshow_matrix(filters, 'Projection Matrix ICA')

# Show filter grating 
opt_frq, opt_ang = STATISTICS.filter_frequency_and_angle(ica.projection_matrix[:,0:20])
VISUALIZATION.imshow_filter_tuning_curve(ica.projection_matrix[:,0:20])
VISUALIZATION.imshow_filter_frequency_angle_histogram(opt_frq, opt_ang)
VISUALIZATION.imshow_filter_optimal_gratings(ica.projection_matrix[:,0:20], opt_frq, opt_ang)

VISUALIZATION.show() 