from mitsuba.core import *
from mitsuba.core import Transform as tf

from head import Head
from arm import Arm
from torso import Torso
from leg import Leg

class Body:
    def __init__(self, torso_center, lower_body_flag):
        self._body_center = torso_center
        self._lower_body_flag = lower_body_flag
        self.head = Head(Point(torso_center.x, torso_center.y+15, torso_center.z))
        self.torso = Torso(Point(torso_center.x, torso_center.y, torso_center.z))
        self.left_arm = Arm(Point(torso_center.x+6.6, torso_center.y+8, torso_center.z))
        self.right_arm = Arm(Point(torso_center.x-6.6, torso_center.y+8, torso_center.z))

        self.right_arm.set_slocal_rotate_angle(180, 0, 0)


        if self._lower_body_flag:
            self.left_leg = Leg(Point(torso_center.x+4, torso_center.y-10.6, torso_center.z))
            self.right_leg = Leg(Point(torso_center.x-4, torso_center.y-10.6, torso_center.z))
            self.right_leg.set_hlocal_rotate_angle(180, 0, 0)

    def horiz_move(self, gx, gy, gz):
        movement = tf.translate(Vector(gx, gy, gz)) # Affine matrix for horizontal movement.
        self._body_center = movement * self._init_torso_center

        self.head.horiz_move(x, y, z)
        self.torso.horiz_move(x, y, z)
        self.left_arm.horiz_move(x, y, z)
        self.right_arm.horiz_move(x, y, z)

        if self._lower_body_flag:
            self.left_leg.horiz_move(x, y, z)
            self.right_leg.horiz_move(x, y, z)

    def calc_body_center(self):
        return self._body_center

    def get_property_list(self):
        property_list = []
        property_list.extend(self.head.get_property_list())
        property_list.extend(self.torso.get_property_list())
        property_list.extend(self.left_arm.get_property_list())
        property_list.extend(self.right_arm.get_property_list())

        if self._lower_body_flag:
            property_list.extend(self.left_leg.get_property_list())
            property_list.extend(self.right_leg.get_property_list())

        return property_list

    def constr_json_data(self):
        # Construcat JSON data
        json_data_head = self.head.constr_json_data()
        json_data_torso = self.torso.constr_json_data()
        json_data_left_arm = self.left_arm.constr_json_data()
        json_data_right_arm = self.right_arm.constr_json_data()
        
        if self._lower_body_flag:
            json_data_left_leg = self.left_leg.constr_json_data()
            json_data_right_leg= self.right_leg.constr_json_data()

        json_data = {
            'joint_angle': {
                'left_shoulder': json_data_left_arm['joint_angle']['shoulder'],
                'left_elbow': json_data_left_arm['joint_angle']['elbow'],
                'right_shoulder': json_data_right_arm['joint_angle']['shoulder'],
                'right_elbow': json_data_right_arm['joint_angle']['elbow']

            },
            'true_position': {
                'head': json_data_head['true_position']['head'],
                'torso': json_data_torso['true_position']['torso'],
                'left_shouler': json_data_left_arm['true_position']['shoulder'],
                'left_elbow': json_data_left_arm['true_position']['elbow'],
                'left_hand': json_data_left_arm['true_position']['hand'],
                'right_shoulder': json_data_right_arm['true_position']['shoulder'],
                'right_elbow': json_data_right_arm['true_position']['elbow'],
                'right_hand': json_data_right_arm['true_position']['hand']
            }
        }

        if self._lower_body_flag:
            json_data['joint_angle']['left_hip'] = json_data_left_leg['joint_angle']['hip']
            json_data['joint_angle']['left_knee'] = json_data_left_leg['joint_angle']['knee']
            json_data['joint_angle']['right_hip'] = json_data_right_leg['joint_angle']['hip']
            json_data['joint_angle']['right_knee'] = json_data_right_leg['joint_angle']['knee']
            json_data['true_position']['left_hip'] = json_data_left_leg['true_position']['hip']
            json_data['true_position']['left_knee'] = json_data_left_leg['true_position']['knee']
            json_data['true_position']['left_foot'] = json_data_left_leg['true_position']['foot']
            json_data['true_position']['right_hip'] = json_data_right_leg['true_position']['hip']
            json_data['true_position']['right_knee'] = json_data_right_leg['true_position']['knee']
            json_data['true_position']['right_foot'] = json_data_right_leg['true_position']['foot']

        return json_data

    def set_from_json_data(self, json_data):
        # Set angle parameters from json data (not json file!)
        self.left_arm.set_from_json_data(json_data['joint_angle']['left_shoulder'], json_data['joint_angle']['left_elbow'])
        self.right_arm.set_from_json_data(json_data['joint_angle']['right_shoulder'], json_data['joint_angle']['right_elbow'])

        if self._lower_body_flag:
            self.left_leg.set_from_json_data(json_data['joint_angle']['left_hip'], json_data['joint_angle']['left_knee'])
            self.right_leg.set_from_json_data(json_data['joint_angle']['right_hip'], json_data['joint_angle']['right_knee'])

    def is_collision_body_parts(self):

        def collision_detection_generator():
            # left arm vs. cranicm
            yield self.left_arm.is_collision_to_sphere(self.head.calc_cranicm_point(), self.head.get_cranicm_radius())

            # left arm vs. torso
            yield self.left_arm.is_collision_to_capsule(self.torso.calc_torso_upper_point(), self.torso.calc_torso_lower_point(), self.torso.get_torso_radius())

            # left arm vs. right upperarm
            yield self.left_arm.is_collision_to_capsule(self.right_arm.calc_shoulder_point(), self.right_arm.calc_upperarm_lower_point(), self.right_arm.get_upperarm_radius())

            # left arm vs. right elbow
            yield self.left_arm.is_collision_to_sphere(self.right_arm.calc_elbow_point(), self.right_arm.get_elbow_radius())

            # left arm vs. right forearm
            yield self.left_arm.is_collision_to_capsule(self.right_arm.calc_forearm_upper_point(), self.right_arm.calc_hand_point(), self.right_arm.get_forearm_radius())

            # right arm vs. cranicm
            yield self.right_arm.is_collision_to_sphere(self.head.calc_cranicm_point(), self.head.get_cranicm_radius())

            # right arm vs. torso
            yield self.right_arm.is_collision_to_capsule(self.torso.calc_torso_upper_point(), self.torso.calc_torso_lower_point(), self.torso.get_torso_radius())

            if self._lower_body_flag:
                # left arm vs. left thigh
                yield self.left_arm.is_collision_to_capsule(self.left_leg.calc_hip_point(), self.left_leg.calc_thigh_lower_point(), self.left_leg.get_thigh_radius())

                # left arm vs. left knee
                yield self.left_arm.is_collision_to_sphere(self.left_leg.calc_knee_point(), self.left_leg.get_knee_radius())

                # left arm vs. left calf
                yield self.left_arm.is_collision_to_capsule(self.left_leg.calc_calf_upper_point(), self.left_leg.calc_foot_point(), self.left_leg.get_calf_radius())

                # left arm vs. right thigh
                yield self.left_arm.is_collision_to_capsule(self.right_leg.calc_hip_point(), self.right_leg.calc_thigh_lower_point(), self.right_leg.get_thigh_radius())

                # left arm vs. right knee
                yield self.left_arm.is_collision_to_sphere(self.right_leg.calc_knee_point(), self.right_leg.get_knee_radius())

                # left arm vs. right calf
                yield self.left_arm.is_collision_to_capsule(self.right_leg.calc_calf_upper_point(), self.right_leg.calc_foot_point(), self.right_leg.get_calf_radius())

                # right arm vs. left thigh
                yield self.right_arm.is_collision_to_capsule(self.left_leg.calc_hip_point(), self.left_leg.calc_thigh_lower_point(), self.left_leg.get_thigh_radius())

                # right arm vs. left thigh
                yield self.right_arm.is_collision_to_capsule(self.left_leg.calc_hip_point(), self.left_leg.calc_thigh_lower_point(), self.left_leg.get_thigh_radius())

                # right arm vs. left knee
                yield self.right_arm.is_collision_to_sphere(self.left_leg.calc_knee_point(), self.left_leg.get_knee_radius())

                # right arm vs. left calf
                yield self.right_arm.is_collision_to_capsule(self.left_leg.calc_calf_upper_point(), self.left_leg.calc_foot_point(), self.left_leg.get_calf_radius())

                # right arm vs. right thigh
                yield self.right_arm.is_collision_to_capsule(self.right_leg.calc_hip_point(), self.right_leg.calc_thigh_lower_point(), self.right_leg.get_thigh_radius())

                # right arm vs. right knee
                yield self.right_arm.is_collision_to_sphere(self.right_leg.calc_knee_point(), self.right_leg.get_knee_radius())

                # right arm vs. right calf
                yield self.right_arm.is_collision_to_capsule(self.right_leg.calc_calf_upper_point(), self.right_leg.calc_foot_point(), self.right_leg.get_calf_radius())

                # left leg vs. cranicm
                yield self.left_leg.is_collision_to_sphere(self.head.calc_cranicm_point(), self.head.get_cranicm_radius())

                # left leg vs. right thigh
                yield self.left_leg.is_collision_to_capsule(self.right_leg.calc_hip_point(), self.right_leg.calc_thigh_lower_point(), self.right_leg.get_thigh_radius())

                # left leg vs. right knee
                yield self.left_leg.is_collision_to_sphere(self.right_leg.calc_knee_point(), self.right_leg.get_knee_radius())

                # left leg vs. right calf
                yield self.left_leg.is_collision_to_capsule(self.right_leg.calc_calf_upper_point(), self.right_leg.calc_foot_point(), self.right_leg.get_calf_radius())

                # right leg vs. cranicm
                yield self.right_leg.is_collision_to_sphere(self.head.calc_cranicm_point(), self.head.get_cranicm_radius())

                # torso vs. left knee
                yield self.torso.is_collision_to_sphere(self.left_leg.calc_knee_point(), self.left_leg.get_knee_radius())

                # torso vs. left calf
                yield self.torso.is_collision_to_capsule(self.left_leg.calc_calf_upper_point(), self.left_leg.calc_foot_point(), self.left_leg.get_calf_radius())

                # torso vs. right knee
                yield self.torso.is_collision_to_sphere(self.right_leg.calc_knee_point(), self.right_leg.get_knee_radius())

                # torso vs. right calf
                yield self.torso.is_collision_to_capsule(self.right_leg.calc_calf_upper_point(), self.right_leg.calc_foot_point(), self.right_leg.get_calf_radius())

        for detection in collision_detection_generator():
            if detection:
                return True

        return False
