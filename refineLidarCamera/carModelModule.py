import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from UlitysModule import Cube, draw_axes
from labelManager import Label3D

class CarSettings:
    def __init__(self, width, height, length, R_Base_to_Lidar, t_Base_to_Lidar, R_Center_to_Base, t_Center_to_Base, R_Basefootprint_to_lidar, t_Basefootprint_to_lidar):
        self.width = width
        self.height = height
        self.length = length

        # transform
        self.R_Base_to_Lidar = R_Base_to_Lidar.astype(np.float64).reshape(3, 3) 
        self.t_Base_to_Lidar = t_Base_to_Lidar.astype(np.float64).reshape(3, 1)
        self.R_Center_to_Base = R_Center_to_Base.astype(np.float64).reshape(3, 3)
        self.t_Center_to_Base = t_Center_to_Base.astype(np.float64).reshape(3, 1)
        self.R_Basefootprint_to_lidar = R_Basefootprint_to_lidar.astype(np.float64).reshape(3, 3)
        self.t_Basefootprint_to_lidar = t_Basefootprint_to_lidar.astype(np.float64).reshape(3, 1)

        self.car_size = [length, width, height]

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_length(self):
        return self.length

class CarModel:
    def __init__(self, car_settings: CarSettings):
        self.car_settings = car_settings
        self.car_size = car_settings.car_size
        self.Base_in_Lidar = None
        self.Center_in_Base = None
        self.Basefootprint_in_Lidar = None
        self.precompute_car_model()

    def precompute_car_model(self):

        # draw the aaxes of the base to the lidar frame, inverse to express in lidar frame
        R_Base_in_Lidar = self.car_settings.R_Base_to_Lidar.T
        t_Base_in_Lidar = (-R_Base_in_Lidar @ self.car_settings.t_Base_to_Lidar).ravel()

        self.Base_in_Lidar = np.identity(4, dtype=np.float64)
        self.Base_in_Lidar[:3, :3] = R_Base_in_Lidar
        self.Base_in_Lidar[:3, 3] = t_Base_in_Lidar

        # draw the aaxes of the center to the base frame, inverse to express in base frame
        R_Center_in_Base = self.car_settings.R_Center_to_Base.T
        t_Center_in_Base = (-R_Center_in_Base @ self.car_settings.t_Center_to_Base).ravel()

        self.Center_in_Base = np.identity(4, dtype=np.float64)
        self.Center_in_Base[:3, :3] = R_Center_in_Base
        self.Center_in_Base[:3, 3] = t_Center_in_Base

        # draw the aaxes of the base footprint to the lidar frame, inverse to express in lidar frame
        R_Basefootprint_in_Lidar = self.car_settings.R_Basefootprint_to_lidar.T
        t_Basefootprint_in_Lidar = (-R_Basefootprint_in_Lidar @ self.car_settings.t_Basefootprint_to_lidar).ravel()

        self.Basefootprint_in_Lidar = np.identity(4, dtype=np.float64)
        self.Basefootprint_in_Lidar[:3, :3] = R_Basefootprint_in_Lidar
        self.Basefootprint_in_Lidar[:3, 3] = t_Basefootprint_in_Lidar


    def draw_axes(self, cube=None, lidar_frame=None, set_model_color=None, identity=None):
        if lidar_frame is None:
            lidar_frame = np.identity(4, dtype=np.float64)

        Base_world = (lidar_frame @ self.Base_in_Lidar).astype(np.float32)

        draw_axes(cube, set_model_color, Base_world)

        Center_world = (Base_world @ self.Center_in_Base).astype(np.float32)

        draw_axes(cube, set_model_color, Center_world)

        Basefootprint_world = (lidar_frame @ self.Basefootprint_in_Lidar).astype(np.float32)

        draw_axes(cube, set_model_color, Basefootprint_world)

        # Extract center + yaw
        center = Center_world[:3, 3]

        R = Center_world[:3, :3]
        yaw = math.atan2(R[1, 0], R[0, 0])

        car_label = Label3D(center, self.car_size, yaw, label_type="car")

        model = car_label.model_matrix().astype(np.float32)

        r, g, b = car_label.color()
        set_model_color(model, r, g, b, 1.0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        cube.draw()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    
    def get_basefootprint_frame(self, lidar_frame=None):
        if lidar_frame is None:
            lidar_frame = np.identity(4, dtype=np.float64)

        return lidar_frame @ self.Basefootprint_in_Lidar