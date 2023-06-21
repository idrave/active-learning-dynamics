from bosdyn.api.robot_state_pb2 import RobotState
from bosdyn.client.frame_helpers import get_vision_tform_body, get_odom_tform_body

class SpotState:
    """Interface for accessing easily relevant bosdyn RobotState fields"""
    def __init__(self, state: RobotState) -> None:
        self.state = state
    
    @property
    def velocity_of_body_in_vision(self):
        linear_velocity = self.state.kinematic_state.velocity_of_body_in_vision.linear
        angular_velocity = self.state.kinematic_state.velocity_of_body_in_vision.angular
        return linear_velocity.x, linear_velocity.y, linear_velocity.z, angular_velocity.x, angular_velocity.y, angular_velocity.z
    
    @property
    def velocity_of_body_in_odom(self):
        linear_velocity = self.state.kinematic_state.velocity_of_body_in_odom.linear
        angular_velocity = self.state.kinematic_state.velocity_of_body_in_odom.angular
        return linear_velocity.x, linear_velocity.y, linear_velocity.z, angular_velocity.x, angular_velocity.y, angular_velocity.z
    
    @property
    def pose_of_body_in_vision(self):
        vision_body_pose = get_vision_tform_body(self.state.kinematic_state.transforms_snapshot)
        return (vision_body_pose.position.x, vision_body_pose.position.y, vision_body_pose.position.z,
                vision_body_pose.rotation.x, vision_body_pose.rotation.y, vision_body_pose.rotation.z, vision_body_pose.rotation.w)

    @property
    def pose_of_body_in_odom(self):
        odom_body_pose = get_odom_tform_body(self.state.kinematic_state.transforms_snapshot)
        return (odom_body_pose.position.x, odom_body_pose.position.y, odom_body_pose.position.z,
                odom_body_pose.rotation.x, odom_body_pose.rotation.y, odom_body_pose.rotation.z, odom_body_pose.rotation.w)
    
    @property
    def joint_states(self):
        return self.state.kinematic_state.joint_states
    
    def to_str(self) -> str:
        s = ""
        for jointState in self.joint_states:
            if jointState.name[:3] != "arm":
                s += "joint_states {\n"
                s += str(jointState)
                s += "}\n"

        s += "velocity_of_body_in_vision {\n"
        s += str(self.state.kinematic_state.velocity_of_body_in_vision)
        s += "}\n"

        s += "velocity_of_body_in_odom {\n"
        s += str(self.state.kinematic_state.velocity_of_body_in_odom)
        s += "}\n"

        odom_body_pose = self.pose_of_body_in_odom
        s += "odom_tform_body {\n"

        s += "position {\n"
        s += "\tx: {:.5f}\n".format(odom_body_pose[0])
        s += "\ty: {:.5f}\n".format(odom_body_pose[1])
        s += "\tz: {:.5f}\n".format(odom_body_pose[2])
        s += "}\n"

        s += "rotation {\n"
        s += "\tw: {:.5f}\n".format(odom_body_pose[6])
        s += "\tx: {:.5f}\n".format(odom_body_pose[3])
        s += "\ty: {:.5f}\n".format(odom_body_pose[4])
        s += "\tz: {:.5f}\n".format(odom_body_pose[5])
        s += "}\n"

        s += "}\n"

        vision_body_pose = self.pose_of_body_in_vision
        s += "vision_tform_body {\n"

        s += "position {\n"
        s += "\tx: {:.5f}\n".format(vision_body_pose[0])
        s += "\ty: {:.5f}\n".format(vision_body_pose[1])
        s += "\tz: {:.5f}\n".format(vision_body_pose[2])
        s += "}\n"

        s += "rotation {\n"
        s += "\tw: {:.5f}\n".format(vision_body_pose[6])
        s += "\tx: {:.5f}\n".format(vision_body_pose[3])
        s += "\ty: {:.5f}\n".format(vision_body_pose[4])
        s += "\tz: {:.5f}\n".format(vision_body_pose[5])
        s += "}\n"

        s += "}\n"
        return s