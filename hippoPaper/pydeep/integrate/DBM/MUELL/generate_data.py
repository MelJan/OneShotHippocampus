import numpy as numx

import pydeep.misc.visualization as VIS


def load_Van_Hateren_Image(path):
    import array
    fin = open( path, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('H', s)
    arr.byteswap()
    return numx.array(arr, dtype='uint16').reshape(1024,1536)



def create_shift_trajectory(image,  patch_width = 16, patch_height = 16, num_sequences = 1, num_samples = 16, step_size = 1):
    width = image.shape[0]
    height = image.shape[1]
    sequence = numx.empty((num_sequences,num_samples,patch_width,patch_height))
    for i in range(num_sequences):
        x = numx.random.randint(0,width-patch_width-num_samples-step_size)
        y = numx.random.randint(0,height-patch_height-num_samples-step_size)
        for j in range(num_samples):
            sequence[i,j] = image[x+j:x+j+patch_width,y+j:y+j+patch_height] 
    return sequence

image = load_Van_Hateren_Image('../../../workspacePy/data/imk00001.imc')
test = create_shift_trajectory(image)
print test.shape
VIS.imshow_matrix(VIS.tile_matrix_columns(test[0].reshape(50,16*16), 16, 16, 1, 20, 1, False), "")
VIS.show()

            