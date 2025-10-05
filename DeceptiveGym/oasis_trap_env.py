import numpy

import DeceptiveGym

import time

import cv2


from pathlib import Path




class OasisTrap:

    def __init__(self, n_envs, max_steps = 256):

        self.n_envs     = n_envs
        self.max_steps  = max_steps

        self.px     = numpy.zeros((n_envs, ), dtype=int)
        self.py     = numpy.zeros((n_envs, ), dtype=int)
        self.steps  = numpy.zeros((n_envs, ), dtype=int)

        self.total_steps    = 0

        package_dir = Path(__file__).resolve().parent

     
        root_dir = str(package_dir) + "/textures/"

        texture_height = 8
        texture_width  = 8

        
        
        self.textures, self.texture_dict = DeceptiveGym.load_textures(root_dir, texture_height, texture_width)

        self.level_map, self.texture_mapping = self._create_level()

        player_texture_idx = self.texture_dict["player"][0]
        self.player_texture = self.textures[player_texture_idx]

        self.background = DeceptiveGym.make_background(self.level_map, self.texture_mapping, self.textures, self.texture_dict)

        self.visited_map = numpy.zeros(self.level_map.shape, dtype=int)

        height = self.level_map.shape[0]
        width  = self.level_map.shape[1]

        self.observation_shape = (3, texture_height*height, texture_width*width)
        self.actions_count     = 5

        self.info = {}
        self.info["negative"]       = 0
        self.info["distractive"]    = 0
        self.info["optimal"]        = 0
       
    def __len__(self):
        return self.n_envs


    def reset(self, env_id):
        self.px[env_id]     = 0
        self.py[env_id]     = 0
        self.steps[env_id]  = 0

        self.states = self._make_state(self.background, self.player_texture, self.px, self.py)
        self.infos  = self._make_infos()

        return self.states[env_id], self.infos[env_id]
    

    def reset_all(self):
        self.px[:]     = 0
        self.py[:]     = 0
        self.steps[:]  = 0  

        self.states = self._make_state(self.background, self.player_texture, self.px, self.py)
        self.infos  = self._make_infos()

        return self.states, self.infos


    def step(self, actions):

        self.steps+= 1

        dx = 0
        dx+= 1*(actions == 1)
        dx+= -1*(actions == 2)
        
        dy = 0
        dy+= 1*(actions == 3)
        dy+=-1*(actions == 4)

        px_new = numpy.clip(self.px + dx, 0, self.level_map.shape[1]-1) 
        py_new = numpy.clip(self.py + dy, 0, self.level_map.shape[0]-1)


        # walls detection
        for n in range(self.n_envs):
            if self.level_map[py_new[n]][px_new[n]] != 1:
                self.px[n] = px_new[n]
                self.py[n] = py_new[n]

        # make states
        
        self.states = self._make_state(self.background, self.player_texture, self.px, self.py)


        rewards = numpy.zeros((self.n_envs, ), dtype=numpy.float32)

        # max steps detected
        dones = self.steps >= self.max_steps


        for n in range(self.n_envs):
            self.visited_map[self.py[n]][self.px[n]]+= 1



        for n in range(self.n_envs):
            # suboptimal goal reached
            if self.px[n] == 2 and self.py[n] == 2:
                dones[n] = True
                rewards[n] = 0.1
                self.info["distractive"]+= 1
            
            # main goal reached
            if self.px[n] == 6 and self.py[n] == 6:
                dones[n] = True
                rewards[n] = 1.0
                self.info["optimal"]+= 1

            # fire fall
            field = self.level_map[self.py[n]][self.px[n]]
            if field == 4:
                dones[n] = True
                rewards[n] = -1.0
                self.info["negative"]+= 1

        self.info["visited_map"] = self.visited_map


        self.infos = self._make_infos()


        return self.states, rewards, dones, self.infos


    def render(self, env_id = 0):
        img = numpy.moveaxis(self.states[env_id], 0, 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("img", img)
        cv2.waitKey(1)

    def _create_level(self):
        level = [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 2, 2, 2, 1, 0, 4, 0],
            [0, 2, 3, 2, 1, 0, 4, 0],
            [0, 2, 2, 2, 0, 0, 4, 0],
            [1, 1, 1, 0, 4, 4, 0, 0],
            [0, 0, 0, 0, 4, 2, 2, 4],
            [0, 4, 4, 4, 0, 2, 3, 2],
            [0, 0, 0, 0, 0, 4, 2, 2]
        ]


        level = numpy.array(level, dtype=int)


        texture_mapping = {}

        texture_mapping[0] = ["desert", 0]
        texture_mapping[1] = ["wall", 1]
        texture_mapping[2] = ["grass", 0]
        texture_mapping[3] = ["water", 0]
        texture_mapping[4] = ["fire", 0]

        return level, texture_mapping


    def _make_state(self, background, player_texture, px, py):

        h = player_texture.shape[1]
        w = player_texture.shape[2]
        tile_size = player_texture.shape[1]

        self.states = numpy.zeros((self.n_envs, ) + background.shape, dtype=numpy.float32)

        self.states[:] = background

        for n in range(self.n_envs):
            py_ofs = py[n]*tile_size
            px_ofs = px[n]*tile_size
            self.states[n, :, py_ofs:py_ofs+h, px_ofs:px_ofs+w] = player_texture

        return self.states
    

    def _make_infos(self):
        infos = []
        for n in range(self.n_envs):
            infos.append(self.info)

        return infos


