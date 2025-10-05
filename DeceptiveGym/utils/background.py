import numpy


def make_background(level, texture_mapping, textures, texture_dict):

    level_height    = level.shape[0]
    level_width     = level.shape[1]

    texture_height = textures[0].shape[1]
    texture_width  = textures[0].shape[2]
    
    res_height = level_height*texture_height
    res_width  = level_width*texture_width

    result = numpy.zeros((3, res_height, res_width))


    for j in range(level_height):
        for i in range(level_width):
            texture_type = level[j][i]
            texture_name, texture_idx = texture_mapping[texture_type]

            tmp = texture_dict[texture_name][texture_idx]

            texture_image = textures[tmp]

            result[:, j*texture_height:(j+1)*texture_height, i*texture_width:(i+1)*texture_width] = texture_image

    result = numpy.array(result, dtype=numpy.float32)
    
    return result
