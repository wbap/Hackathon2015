from mitsuba.core import *
from mitsuba.core import Transform as tf

from coll_detection import *

class Leg:
    def __init__(self, ihp): # ihp is initial hip point
        self._thigh_radius = 1.5
        self._thigh_length = 8.0
        self._knee_radius = 1.5
        self._calf_radius = 1.5
        self._calf_length = 8.0

        self._init_hip_point = ihp # == _init_thigh_start_point
        self._init_thigh_end_point = Point(ihp.x, ihp.y-self._thigh_length, ihp.z)
        self._init_thigh_toWorld = tf.translate(Vector(ihp.x, ihp.y-(self._thigh_length/2), ihp.z)) * tf.scale(Vector(1, self._thigh_length, 1)) * tf.rotate(Vector(1, 0, 0), -90) * tf.translate(Vector(0, 0, -0.5))
        self._init_knee_point = Point(ihp.x, ihp.y-(self._thigh_length+self._thigh_radius+self._knee_radius), ihp.z)
        self._init_calf_start_point = Point(ihp.x, ihp.y-(self._thigh_length+self._thigh_radius+self._knee_radius*2+self._calf_radius), ihp.z)
        self._init_foot_point = Point(ihp.x, ihp.y-(self._thigh_length+self._thigh_radius+self._knee_radius*2+self._calf_radius+self._calf_length), ihp.z) # == _init_calf_end_point
        self._init_calf_toWorld = tf.translate(Vector(ihp.x, ihp.y-(self._thigh_length+self._thigh_radius+self._knee_radius*2+self._calf_radius+self._calf_length/2), ihp.z)) * tf.scale(Vector(1, self._calf_length, 1)) * tf.rotate(Vector(1, 0, 0), -90) * tf.translate(Vector(0, 0, -0.5))

        # local axes at hip
        self._hlocal_axes = {
                'x' : Vector(1, 0, 0),
                'y' : Vector(0, 1, 0),
                'z' : Vector(0, 0, 1)
                }
        # rotate angle of local axes at hip
        self._hlocal_axes_rotate_angle = {
                'pitch' : 0,
                'yaw' : 0,
                'roll' : 0
                }
        
        self._hip_pitch = 0
        self._hip_yaw = 0
        self._hip_roll = 0
        self._knee_angle = 0

        self.hip_joint_prop = {
                'type' : 'sphere', 
                'center' : self._init_hip_point,
                'radius' : self._thigh_radius ,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

        self.thigh_prop = {
                'type' : 'cylinder',
                'toWorld': self._init_thigh_toWorld,
                'radius' : self._thigh_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }
        
        self.thigh_endcap_prop = {
                'type' : 'sphere', 
                'center' : self._init_thigh_end_point,
                'radius' : self._thigh_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

        self.knee_joint_prop  = {
                'type' : 'sphere',
                'center' : self._init_knee_point,
                'radius' : self._knee_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.1, 0.1, 0.1])
                }
            }

        self.calf_endcap_prop = {
                'type' : 'sphere', 
                'center' : self._init_calf_start_point,
                'radius' : self._calf_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }


        self.calf_prop = {
                'type' : 'cylinder',
                'toWorld' : self._init_calf_toWorld,
                'radius' : self._calf_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }
        self.foot_prop = {
                'type' : 'sphere',
                'center' : self._init_foot_point,
                'radius' : self._calf_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

    def get_curt_hip_angles(self):
        return [self._hip_pitch, self._hip_yaw, self._hip_yaw]

    def get_curt_knee_angle(self):
        return self._knee_angle

    def get_curt_hlocal_axes(self):
        return self._hlocal_axes

    def get_curt_hlocal_rotate_angle(self):
        return self._hlocal_axes_rotate_angle
    
    def set_hlocal_rotate_angle(self, gpitch, gyaw, groll):
        self._hlocal_axes_rotate_angle['pitch'] = gpitch
        self._hlocal_axes_rotate_angle['yaw'] = gyaw
        self._hlocal_axes_rotate_angle['roll'] = groll
        rotation = tf.rotate(Vector(1, 0, 0), gpitch) * tf.rotate(Vector(0, 1, 0), gyaw) * tf.rotate(Vector(0, 0, 1), groll)
        self._hlocal_axes['x'] = rotation * self._hlocal_axes['x'] 
        self._hlocal_axes['y'] = rotation * self._hlocal_axes['y'] 
        self._hlocal_axes['z'] = rotation * self._hlocal_axes['z']

        self.set_joint_angles(self._hip_pitch, self._hip_yaw, self._hip_yaw, self._knee_angle)

        return self._hlocal_axes

    def _create_xyz_rotation(self, pitch, yaw, roll):
        return tf.rotate(self._hlocal_axes['x'], pitch) * tf.rotate(self._hlocal_axes['y'], yaw) * tf.rotate(self._hlocal_axes['z'], roll)

    def set_joint_angles(self, hip_pitch, hip_yaw, hip_roll, knee_angle):
        self._hip_pitch = hip_pitch
        self._hip_yaw = hip_yaw
        self._hip_roll = hip_roll
        self._knee_angle = knee_angle

        # sto is the translate "hip to origin"
        sto = tf.translate(Vector(-self.hip_joint_prop['center'][0], -self.hip_joint_prop['center'][1], -self.hip_joint_prop['center'][2]))
        # ots is the translate "hip from origin"
        ots = tf.translate(Vector(self.hip_joint_prop['center'][0], self.hip_joint_prop['center'][1], self.hip_joint_prop['center'][2]))
        self.thigh_prop['toWorld'] = ots * self._create_xyz_rotation(hip_pitch, hip_yaw, hip_roll) * sto * self._init_thigh_toWorld
        self.thigh_endcap_prop['center'] = ots * self._create_xyz_rotation(hip_pitch, hip_yaw, hip_roll) * sto * self._init_thigh_end_point
        self.knee_joint_prop['center'] = ots * self._create_xyz_rotation(hip_pitch, hip_yaw, hip_roll) * sto * self._init_knee_point
        self.calf_endcap_prop['center'] = ots * self._create_xyz_rotation(hip_pitch, hip_yaw, hip_roll) * sto * self._init_calf_start_point
        self.calf_prop['toWorld'] =  ots * self._create_xyz_rotation(hip_pitch, hip_yaw, hip_roll) * sto * self._init_calf_toWorld
        self.foot_prop['center'] = ots * self._create_xyz_rotation(hip_pitch, hip_yaw, hip_roll) * sto * self._init_foot_point

        # eto is the translate "knee to origin"
        eto = tf.translate(Vector(-self.knee_joint_prop['center'][0], -self.knee_joint_prop['center'][1], -self.knee_joint_prop['center'][2])) 
        # ots is the translate "origin to hip"
        ote = tf.translate(Vector(self.knee_joint_prop['center'][0], self.knee_joint_prop['center'][1], self.knee_joint_prop['center'][2]))
        knee_axis = self.thigh_prop['toWorld'] * self._hlocal_axes['x']
        self.calf_endcap_prop['center'] = ote * tf.rotate(knee_axis, knee_angle) * eto * self.calf_endcap_prop['center']
        self.calf_prop['toWorld'] = ote * tf.rotate(knee_axis, knee_angle) * eto * self.calf_prop['toWorld']
        self.foot_prop['center'] = ote * tf.rotate(knee_axis, knee_angle) * eto * self.foot_prop['center']
    
    def rotate_joint_angles(self, hip_pitch, hip_yaw, hip_roll, knee_angle):
        self.set_joint_angles(self._hip_pitch + hip_pitch, self._hip_yaw + hip_yaw, self._hip_roll + hip_roll, self._knee_angle + knee_angle)

    def set_hip_angle(self, pitch, yaw, roll):
        self.set_joint_angles(pitch, yaw, roll, self._knee_angle)

    def rotate_hip(self, pitch, yaw, roll):
        self.set_hip_angle(self._hip_pitch + pitch, self._hip_yaw + yaw, self._hip_roll + roll)

    def set_knee_angle(self, angle):
        self.set_joint_angles(self._hip_pitch, self._hip_yaw, self._hip_roll, angle)
        
    def rotate_knee(self, angle):
        self.set_knee_angle(self._knee_angle + angle)

    def horiz_move(self, gx, gy, gz):
        movement = tf.translate(Vector(gx, gy, gz)) # Affine matrix for horizontal movement.
        self._init_hip_point = movement * self._init_hip_point
        self._init_thigh_toWorld = movement * self._init_thigh_toWorld
        self._init_thigh_end_point = movement * self._init_thigh_end_point 
        self._init_knee_point = movement * self._init_knee_point
        self._init_calf_start_point = movement * self._init_calf_start_point
        self._init_foot_point = movement * self._init_foot_point
        self._init_calf_toWorld = movement * self._init_calf_toWorld

        self.hip_joint_prop['center'] = movement * self.hip_joint_prop['center']
        self.thigh_endcap_prop['center'] = movement * self.thigh_endcap_prop['center']
        self.thigh_prop['toWorld'] = movement * self.thigh_prop['toWorld']
        self.knee_joint_prop['center'] = movement * self.knee_joint_prop['center']
        self.calf_endcap_prop['center'] = movement * self.calf_endcap_prop['center']
        self.calf_prop['toWorld'] = movement * self.calf_prop['toWorld']
        self.foot_prop['center'] = movement * self.foot_prop['center']

    def calc_hip_point(self):
        return self.hip_joint_prop['center']

    def calc_thigh_lower_point(self):
        return self.thigh_endcap_prop['center']

    def calc_knee_point(self):
        return self.knee_joint_prop['center']

    def calc_calf_upper_point(self):
        return self.calf_endcap_prop['center']

    def calc_foot_point(self):
        return self.foot_prop['center']
    
    def get_thigh_radius(self):
        return self._thigh_radius
    
    def get_thigh_length(self):
        return self._thigh_length

    def get_knee_radius(self):
        return self._knee_radius
    
    def get_calf_radius(self):
        return self._calf_radius
    
    def get_calf_length(self):
        return self._calf_length

    def get_property_list(self):
        property_list = []
        property_list.append(self.hip_joint_prop)
        property_list.append(self.thigh_prop)
        property_list.append(self.thigh_endcap_prop)
        property_list.append(self.knee_joint_prop)
        property_list.append(self.calf_endcap_prop)
        property_list.append(self.calf_prop)
        property_list.append(self.foot_prop)

        return property_list

    def constr_json_data(self):
        # Construct JSON data
        json_data = {
            'joint_angle': {
                # 'hip': dict(zip(['p', 'y', 'r'], self.get_curt_hip_angles())),
                'hip': {
                    'p': self._hip_pitch,
                    'y': self._hip_yaw,
                    'r': self._hip_roll,
                },
                'knee': {
                    'p': self.get_curt_knee_angle()
                }
            },
            'true_position': {
                'hip': dict(zip(['x', 'y', 'z'], self.calc_hip_point())),
                'knee': dict(zip(['x', 'y', 'z'], self.calc_knee_point())),
                'foot' : dict(zip(['x', 'y', 'z'], self.calc_foot_point()))
            }
        }
        return json_data
    
    def set_from_json_data(self, json_hip_data, json_knee_data):
        # Set angle parameters from json data (not json file!)
        self.set_joint_angles(json_hip_data['p'], json_hip_data['y'], json_hip_data['r'], json_knee_data['p']) 

    def is_collision_to_capsule(self, p1, p2, radius):

        def collision_detection_generator():
            # vs. thigh
            min_dist_thigh = calc_min_dist_segment_segment(self.calc_hip_point(), self.calc_thigh_lower_point(), p1, p2)
            yield min_dist_thigh <= radius + self.get_thigh_radius()

            # vs. knee
            min_dist_knee = calc_min_dist_segment_point(p1, p2, self.calc_knee_point())
            yield min_dist_knee <= radius + self.get_knee_radius()
            
            # vs. calf
            min_dist_calf = calc_min_dist_segment_segment(self.calc_calf_upper_point(), self.calc_foot_point(), p1, p2)
            yield min_dist_calf <= radius + self.get_calf_radius()

        print 'Now, calculating the collision of leg and capsule'
        
        for detection in collision_detection_generator():
            if detection:
                return True
        
        return False

    def is_collision_to_sphere(self, p1, radius):

        def collision_detection_generator():
            # vs. thigh
            min_dist_thigh = calc_min_dist_segment_point(self.calc_hip_point(), self.calc_thigh_lower_point(), p1)
            yield min_dist_thigh <= radius + self.get_thigh_radius()
            
            # vs. knee
            min_dist_knee = calc_min_dist_point_point(self.calc_knee_point(), p1)
            yield min_dist_knee <= radius + self.get_knee_radius()

            # vs. calf
            min_dist_calf = calc_min_dist_segment_point(self.calc_calf_upper_point(), self.calc_foot_point(), p1)
            yield min_dist_calf <= radius + self.get_calf_radius()

        print 'Now, calculating the collision of leg and sphere'

        for detection in collision_detection_generator():
            if detection:
                return True
        
        return False
