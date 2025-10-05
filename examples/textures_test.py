import DeceptiveGym

import numpy


import cv2

if __name__ == "__main__":
    root_dir = "../DeceptiveGym/textures/"

    height = 32
    width  = 32
    textures, texture_dict = DeceptiveGym.load_textures(root_dir, height, width)



    texture_mapping = {}

    texture_mapping[0] = ["desert", 0]
    texture_mapping[1] = ["wall", 1]
    texture_mapping[2] = ["grass", 0]
    texture_mapping[3] = ["water", 0]
    texture_mapping[4] = ["fire", 0]
    texture_mapping[6] = ["player", 0]

    level = numpy.zeros((8, 8), dtype=int)

    # make oasis 1
    level[2][2] = 3
    
    level[1][1] = 2
    level[1][2] = 2
    level[1][3] = 2

    level[2][1] = 2
    level[2][3] = 2

    level[3][1] = 2
    level[3][2] = 2
    level[3][3] = 2


    # make oasis 2
    level[6][6] = 3
    
    level[5][5] = 2
    level[5][6] = 2
    level[5][7] = 2

    level[6][5] = 2
    level[6][7] = 2

    level[7][5] = 2
    level[7][6] = 2
    level[7][7] = 2

    # make wall
    level[4][4] = 1
    level[4][5] = 1
    level[4][7] = 1
    level[5][4] = 1
    level[7][4] = 1

    level[3][4] = 1
    level[2][4] = 1
    level[2][5] = 1
    level[2][6] = 1

    level[0][7] = 1
    level[0][6] = 1
    level[1][4] = 1


    level[4][3] = 1
    level[4][2] = 1
    level[5][2] = 1
    level[6][2] = 1

    level[7][0] = 1
    level[6][0] = 1
    level[4][1] = 1



    level[0][0] = 6


    
    background = make_background(level, texture_mapping, textures, texture_dict)



    img = numpy.moveaxis(background, 0, 2)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    