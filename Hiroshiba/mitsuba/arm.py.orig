from mitsuba.core import *
from mitsuba.core import Transform as tf

from coll_detection import *

class Arm:
    def __init__(self, isp): # isp is initial shoulder point
        self._upperarm_radius = 1.0
        self._upperarm_length = 6.0
        self._elbow_radius = 1.0
        self._forearm_radius = 1.0
        self._forearm_length = 6.0

        self._init_shoulder_point = isp # == _init_upperarm_start_point
        self._init_upperarm_end_point = Point(isp.x, isp.y-self._upperarm_length, isp.z)
        self._init_upperarm_toWorld = tf.translate(Vector(isp.x, isp.y-(self._upperarm_length/2), isp.z)) * tf.scale(Vector(1, self._upperarm_length, 1)) * tf.rotate(Vector(1, 0, 0), -90) * tf.translate(Vector(0, 0, -0.5))
        self._init_elbow_point = Point(isp.x, isp.y-(self._upperarm_length+self._upperarm_radius+self._elbow_radius), isp.z)
        self._init_forearm_start_point = Point(isp.x, isp.y-(self._upperarm_length+self._upperarm_radius+self._elbow_radius*2+self._forearm_radius), isp.z)
        self._init_hand_point = Point(isp.x, isp.y-(self._upperarm_length+self._upperarm_radius+self._elbow_radius*2+self._forearm_radius+self._forearm_length), isp.z) # == _init_forearm_end_point
        self._init_forearm_toWorld = tf.translate(Vector(isp.x, isp.y-(self._upperarm_length+self._upperarm_radius+self._elbow_radius*2+self._forearm_radius+self._forearm_length/2), isp.z)) * tf.scale(Vector(1, self._forearm_length, 1)) * tf.rotate(Vector(1, 0, 0), -90) * tf.translate(Vector(0, 0, -0.5))

        # local axes at shoulder
        self._slocal_axes = {
                'x' : Vector(1, 0, 0),
                'y' : Vector(0, 1, 0),
                'z' : Vector(0, 0, 1)
                }
        # rotate angle of local axes at shoulder
        self._slocal_axes_rotate_angle = {
                'pitch' : 0,
                'yaw' : 0,
                'roll' : 0
                }
        
        self._shoulder_pitch = 0
        self._shoulder_yaw = 0
        self._shoulder_roll = 0
        self._elbow_angle = 0

        self.shoulder_joint_prop = {
                'type' : 'sphere', 
                'center' : self._init_shoulder_point,
                'radius' : self._upperarm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

        self.upperarm_prop = {
                'type' : 'cylinder',
                'toWorld': self._init_upperarm_toWorld,
                'radius' : self._upperarm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }
        
        self.upperarm_endcap_prop = {
                'type' : 'sphere', 
                'center' : self._init_upperarm_end_point,
                'radius' : self._upperarm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

        self.elbow_joint_prop  = {
                'type' : 'sphere',
                'center' : self._init_elbow_point,
                'radius' : self._elbow_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.1, 0.1, 0.1])
                }
            }

        self.forearm_endcap_prop = {
                'type' : 'sphere', 
                'center' : self._init_forearm_start_point,
                'radius' : self._forearm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }


        self.forearm_prop = {
                'type' : 'cylinder',
                'toWorld' : self._init_forearm_toWorld,
                'radius' : self._forearm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }
        self.hand_prop = {
                'type' : 'sphere',
                'center' : self._init_hand_point,
                'radius' : self._forearm_radius,
                'bsdf' : {
                    'type' : 'ward',
                    'alphaU' : 0.003,
                    'alphaV' : 0.003,
                    'specularReflectance' : Spectrum(0.01),
                    'diffuseReflectance' : Spectrum([0.05, 0.05, 0.05])
                }
            }

    def get_curt_shoulder_angles(self):
        return [self._shoulder_pitch, self._shoulder_yaw, self._shoulder_yaw]

    def get_curt_elbow_angle(self):
        return self._elbow_angle

    def get_curt_slocal_axes(self):
        return self._slocal_axes

    def get_curt_slocal_rotate_angle(self):
        return self._slocal_axes_rotate_angle
    
    def set_slocal_rotate_angle(self, gpitch, gyaw, groll):
        self._slocal_axes_rotate_angle['pitch'] = gpitch
        self._slocal_axes_rotate_angle['yaw'] = gyaw
        self._slocal_axes_rotate_angle['roll'] = groll
        rotation = tf.rotate(Vector(1, 0, 0), gpitch) * tf.rotate(Vector(0, 1, 0), gyaw) * tf.rotate(Vector(0, 0, 1), groll)
        self._slocal_axes['x'] = rotation * self._slocal_axes['x'] 
        self._slocal_axes['y'] = rotation * self._slocal_axes['y'] 
        self._slocal_axes['z'] = rotation * self._slocal_axes['z']

        self.set_joint_angles(self._shoulder_pitch, self._shoulder_yaw, self._shoulder_yaw, self._elbow_angle)

        return self._slocal_axes

    def _create_xyz_rotation(self, pitch, yaw, roll):
        return tf.rotate(self._slocal_axes['x'], pitch) * tf.rotate(self._slocal_axes['y'], yaw) * tf.rotate(self._slocal_axes['z'], roll)

    def set_joint_angles(self, shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_angle):
        self._shoulder_pitch = shoulder_pitch
        self._shoulder_yaw = shoulder_yaw
        self._shoulder_roll = shoulder_roll
        self._elbow_angle = elbow_angle

        # sto is the translate "shoulder to origin"
        sto = tf.translate(Vector(-self.shoulder_joint_prop['center'][0], -self.shoulder_joint_prop['center'][1], -self.shoulder_joint_prop['center'][2]))
        # ots is the translate "shoulder from origin"
        ots = tf.translate(Vector(self.shoulder_joint_prop['center'][0], self.shoulder_joint_prop['center'][1], self.shoulder_joint_prop['center'][2]))
        self.upperarm_prop['toWorld'] = ots * self._create_xyz_rotation(shoulder_pitch, shoulder_yaw, shoulder_roll) * sto * self._init_upperarm_toWorld
        self.upperarm_endcap_prop['center'] = ots * self._create_xyz_rotation(shoulder_pitch, shoulder_yaw, shoulder_roll) * sto * self._init_upperarm_end_point
        self.elbow_joint_prop['center'] = ots * self._create_xyz_rotation(shoulder_pitch, shoulder_yaw, shoulder_roll) * sto * self._init_elbow_point
        self.forearm_endcap_prop['center'] = ots * self._create_xyz_rotation(shoulder_pitch, shoulder_yaw, shoulder_roll) * sto * self._init_forearm_start_point
        self.forearm_prop['toWorld'] =  ots * self._create_xyz_rotation(shoulder_pitch, shoulder_yaw, shoulder_roll) * sto * self._init_forearm_toWorld
        self.hand_prop['center'] = ots * self._create_xyz_rotation(shoulder_pitch, shoulder_yaw, shoulder_roll) * sto * self._init_hand_point

        # eto is the translate "elbow to origin"
        eto = tf.translate(Vector(-self.elbow_joint_prop['center'][0], -self.elbow_joint_prop['center'][1], -self.elbow_joint_prop['center'][2])) 
        # ots is the translate "origin to shoulder"
        ote = tf.translate(Vector(self.elbow_joint_prop['center'][0], self.elbow_joint_prop['center'][1], self.elbow_joint_prop['center'][2]))
        elbow_axis = self.upperarm_prop['toWorld'] * (-self._slocal_axes['x'])
        self.forearm_endcap_prop['center'] = ote * tf.rotate(elbow_axis, elbow_angle) * eto * self.forearm_endcap_prop['center']
        self.forearm_prop['toWorld'] = ote * tf.rotate(elbow_axis, elbow_angle) * eto * self.forearm_prop['toWorld']
        self.hand_prop['center'] = ote * tf.rotate(elbow_axis, elbow_angle) * eto * self.hand_prop['center']
    
    def rotate_joint_angles(self, shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_angle):
        self.set_joint_angles(self._shoulder_pitch + shoulder_pitch, self._shoulder_yaw + shoulder_yaw, self._shoulder_roll + shoulder_roll, self._elbow_angle + elbow_angle)

    def set_shoulder_angle(self, pitch, yaw, roll):
        self.set_joint_angles(pitch, yaw, roll, self._elbow_angle)

    def rotate_shoulder(self, pitch, yaw, roll):
        self.set_shoulder_angle(self._shoulder_pitch + pitch, self._shoulder_yaw + yaw, self._shoulder_roll + roll)

    def set_elbow_angle(self, angle):
        self.set_joint_angles(self._shoulder_pitch, self._shoulder_yaw, self._shoulder_roll, angle)
        
    def rotate_elbow(self, angle):
        self.set_elbow_angle(self._elbow_angle + angle)

    def horiz_move(self, gx, gy, gz):
        movement = tf.translate(Vector(gx, gy, gz)) # Affine matrix for horizontal movement.
        self._init_shoulder_point = movement * self._init_shoulder_point
        self._init_upperarm_toWorld = movement * self._init_upperarm_toWorld
        self._init_upperarm_end_point = movement * self._init_upperarm_end_point 
        self._init_elbow_point = movement * self._init_elbow_point
        self._init_forearm_start_point = movement * self._init_forearm_start_point
        self._init_hand_point = movement * self._init_hand_point
        self._init_forearm_toWorld = movement * self._init_forearm_toWorld

        self.shoulder_joint_prop['center'] = movement * self.shoulder_joint_prop['center']
        self.upperarm_endcap_prop['center'] = movement * self.upperarm_endcap_prop['center']
        self.upperarm_prop['toWorld'] = movement * self.upperarm_prop['toWorld']
        self.elbow_joint_prop['center'] = movement * self.elbow_joint_prop['center']
        self.forearm_endcap_prop['center'] = movement * self.forearm_endcap_prop['center']
        self.forearm_prop['toWorld'] = movement * self.forearm_prop['toWorld']
        self.hand_prop['center'] = movement * self.hand_prop['center']

    def calc_shoulder_point(self):
        return self.shoulder_joint_prop['center']

    def calc_upperarm_lower_point(self):
        return self.upperarm_endcap_prop['center']

    def calc_elbow_point(self):
        return self.elbow_joint_prop['center']

    def calc_forearm_upper_point(self):
        return self.forearm_endcap_prop['center']

    def calc_hand_point(self):
        return self.hand_prop['center']

    def get_upperarm_radius(self):
        return self._upperarm_radius

    def get_upperarm_length(self):
        return self._upperarm_length

    def get_elbow_radius(self):
        return self._elbow_radius
    
    def get_forearm_radius(self):
        return self._forearm_radius

    def get_forearm_length(self):
        return self._forearm_length
    
    def get_property_list(self):
        property_list = []
        property_list.append(self.shoulder_joint_prop)
        property_list.append(self.upperarm_prop)
        property_list.append(self.upperarm_endcap_prop)
        property_list.append(self.elbow_joint_prop)
        property_list.append(self.forearm_endcap_prop)
        property_list.append(self.forearm_prop)
        property_list.append(self.hand_prop)

        return property_list

    def constr_json_data(self):
        # Construct JSON data
        json_data = {
            'joint_angle': {
                # 'shoulder': dict(zip(['p', 'y', 'r'], self.get_curt_shoulder_angles())),
                'shoulder': {
                    'p': self._shoulder_pitch,
                    'y': self._shoulder_yaw,
                    'r': self._shoulder_roll
                },
                'elbow': {
                    'p': self.get_curt_elbow_angle()
                }
            },
            'true_position': {
                'shoulder': dict(zip(['x', 'y', 'z'], self.calc_shoulder_point())),
                'elbow': dict(zip(['x', 'y', 'z'], self.calc_elbow_point())),
                'hand' : dict(zip(['x', 'y', 'z'], self.calc_hand_point()))
            }
        }
        return json_data

    def set_from_json_data(self, json_shoulder_data, json_elbow_data):
        # Set angle parameters from json data (not json file!)
        self.set_joint_angles(json_shoulder_data['p'], json_shoulder_data['y'], json_shoulder_data['r'], json_elbow_data['p'])

    def is_collision_to_capsule(self, p1, p2, radius):

        def collision_detection_generator():
            # vs. upperarm
            min_dist_upperarm = calc_min_dist_segment_segment(self.calc_shoulder_point(), self.calc_upperarm_lower_point(), p1, p2)
            yield min_dist_upperarm <= radius + self.get_upperarm_radius()

            # vs. elbow
            min_dist_elbow = calc_min_dist_segment_point(p1, p2, self.calc_elbow_point())
            yield min_dist_elbow <= radius + self.get_elbow_radius()
            
            # vs. forearm
            min_dist_forearm = calc_min_dist_segment_segment(self.calc_forearm_upper_point(), self.calc_hand_point(), p1, p2)
            yield min_dist_forearm <= radius + self.get_forearm_radius()

        print 'Now, calculating the collision of arm and capsule'

        for detection in collision_detection_generator():
            if detection:
                return True
        
        return False

    def is_collision_to_sphere(self, p1, radius):

        def collision_detection_generator():
            # vs. upperarm
            min_dist_upperarm = calc_min_dist_segment_point(self.calc_shoulder_point(), self.calc_upperarm_lower_point(), p1)
            yield min_dist_upperarm <= radius + self.get_upperarm_radius()

            # vs. elbow
            min_dist_elbow = calc_min_dist_point_point(self.calc_elbow_point(), p1)
            yield min_dist_elbow <= radius + self.get_elbow_radius()

             # vs. forearm
            min_dist_forearm = calc_min_dist_segment_point(self.calc_forearm_upper_point(), self.calc_hand_point(), p1)
            yield min_dist_forearm <= radius + self.get_forearm_radius()

        print 'Now, calculating the collision of arm and sphere' 

        for detection in collision_detection_generator():
            if detection:
                return True
        
        return False
