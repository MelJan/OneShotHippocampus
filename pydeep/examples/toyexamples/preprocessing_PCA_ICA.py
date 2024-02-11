''' Example for the preprocessing functions on a 2D example.  
        
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

import numpy as numx
import pydeep.misc.toyproblems as TOY_DATA
import pydeep.misc.visualization as VISUALIZATION
import pydeep.preprocessing as preprocessing

numx.random.seed(422)

# Create data
data, mixing_matrix = TOY_DATA.generate_2d_mixtures(50000, 0, 1.0)

# ZCA
zca = preprocessing.ZCA(data.shape[1])
zca.train(data)
data_zca = zca.project(data)

# ICA
ica = preprocessing.ICA(data_zca.shape[1])
ica.train(data_zca, iterations = 1000)
data_ica = ica.project(data_zca)

# Display results
VISUALIZATION.figure(0, figsize = [5,5])
VISUALIZATION.title("Data")
VISUALIZATION.plot_2d_data(data)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.figure(1, figsize = [5,5])
VISUALIZATION.title("ZCA")
VISUALIZATION.plot_2d_data(data_zca)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.figure(2, figsize = [5,5])
VISUALIZATION.title("ICA")
VISUALIZATION.plot_2d_data(data_ica)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.figure(3, figsize = [5,5])
VISUALIZATION.title("ICA matrix ZCA space")
VISUALIZATION.plot_2d_data(data_zca)
VISUALIZATION.plot_2d_weights(ica.projection_matrix)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.figure(4, figsize = [5,5])
VISUALIZATION.title("ICA matrix data space")
VISUALIZATION.plot_2d_data(data)
VISUALIZATION.plot_2d_weights(zca.unproject(ica.unprojection_matrix).T,zca.mean)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.figure(5, figsize = [5,5])
VISUALIZATION.title("Mixing matrix ZCA space")
VISUALIZATION.plot_2d_data(data_zca)
VISUALIZATION.plot_2d_weights(zca.project(mixing_matrix.T).T)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.figure(6, figsize = [5,5])
VISUALIZATION.title("Mixing matrix data space")
VISUALIZATION.plot_2d_data(data)
VISUALIZATION.plot_2d_weights(zca.project(zca.unproject(mixing_matrix)),zca.mean)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-3,3,-3,3])

VISUALIZATION.show()

'''
# Create data
data, mixing_matrix = TOY_DATA.generate_2d_mixtures(50000, 1, 1.0)

# Rescale data
data_rs = preprocessing.rescale_data(data, -1, 1)

# Binarize data
data_bin = preprocessing.binarize_data(data)

# Remove mean
data_rm = preprocessing.remove_rows_means(data)

# STANDARIZER
std = preprocessing.STANDARIZER(data.shape[1])
std.train_images(data)
data_std = std.project(data)

# ZCA
zca = preprocessing.ZCA(data.shape[1])
zca.train_images(data)
data_zca = zca.project(data)

# PCA
pca = preprocessing.PCA(data.shape[1],whiten=True)
pca.train_images(data)
data_pca = pca.project(data)

# ICA
ica = preprocessing.ICA(data_pca.shape[1])
ica.train_images(data_pca, iterations = 1000)
data_ica = ica.project(data_pca)

# Display results
VISUALIZATION.figure(0, figsize = [5,5])
VISUALIZATION.title("Data")
VISUALIZATION.plot_2d_data(data)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(1, figsize = [5,5])
VISUALIZATION.title("Rescale")
VISUALIZATION.plot_2d_data(data_rs)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-2,2,-2,2])

VISUALIZATION.figure(2, figsize = [5,5])
VISUALIZATION.title("Remove mean")
VISUALIZATION.plot_2d_data(data_rm)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(3, figsize = [5,5])
VISUALIZATION.title("Binarize")
VISUALIZATION.plot_2d_data(data_bin)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-2,2,-2,2])

VISUALIZATION.figure(4, figsize = [5,5])
VISUALIZATION.title("Standarizer")
VISUALIZATION.plot_2d_data(data_std)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(5, figsize = [5,5])
VISUALIZATION.title("PCA")
VISUALIZATION.plot_2d_data(data_pca)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-50,50,-50,50])

VISUALIZATION.figure(6, figsize = [5,5])
VISUALIZATION.title("ZCA")
VISUALIZATION.plot_2d_data(data_zca)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(7, figsize = [5,5])
VISUALIZATION.title("ICA")
VISUALIZATION.plot_2d_data(data_ica)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])


VISUALIZATION.figure(8, figsize = [5,5])
VISUALIZATION.title("PCA weights")
VISUALIZATION.plot_2d_data(data-numx.mean(data,axis = 0))
VISUALIZATION.plot_2d_weights(pca.projection_matrix)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(9, figsize = [5,5])
VISUALIZATION.title("ICA weights in ZCA space")
VISUALIZATION.plot_2d_data(data_zca)
VISUALIZATION.plot_2d_weights(ica.projection_matrix)
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(10, figsize = [5,5])
VISUALIZATION.title("ICA weights")
VISUALIZATION.plot_2d_data(data)
VISUALIZATION.plot_2d_weights(pca.unproject(ica.projection_matrix),pca.mean.reshape(1,2),color='red',bias_color='yellow')
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])

VISUALIZATION.figure(11, figsize = [5,5])
VISUALIZATION.title("mixing data weights")
print zca.mean
VISUALIZATION.plot_2d_data(data)
VISUALIZATION.plot_2d_weights(mixing_matrix,pca.mean.reshape(1,2),color='red',bias_color='yellow')
VISUALIZATION.axis('equal')
VISUALIZATION.axis([-10,10,-10,10])


VISUALIZATION.show()
'''