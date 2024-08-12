# make sure you are in the devel/setup.bash environment or else no messages for you
try:
    import rospy
    from nav_msgs.msg import Odometry,Path
except ModuleNotFoundError:
    print("WARNING: ROS ENV NOT SET. "
          "Make sure you are in the devel/setup.bash environment or else no messages for you")
    Markers = None
import numpy as np
from time import sleep

class RigTopicListener:
    # TODO finish this class
    def __init__(self):
        self.data = Odometry()

    def listen_start(self):
        # Initialize the node, needs to be done only once
        rospy.init_node('listener', anonymous=True)

        # subscribe to relevant topic and tap into the right message type published by the topic
        rospy.Subscriber("camera/odom/sample", Odometry, self.callback)
        print('Rig is initiated...')

        # run until rospy connection is broken
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
            break
        # rospy.spin()
        # spin() simply keeps python from exiting until this node is stopped

    def callback(self, data):
        self.data = data
        return self.data

    def get_sensory_observation(self):
        """
        Extraction of sensory data (angular velocity and speed)
        :return:
        """
        position_heading = [
            self.data.pose.pose.position.x,     # 0
            self.data.pose.pose.position.y,
            self.data.pose.pose.position.z,
            self.data.pose.pose.orientation.x,  # 3
            self.data.pose.pose.orientation.y,
            self.data.pose.pose.orientation.z,
            self.data.pose.pose.orientation.w,
            self.data.twist.twist.linear.x,     # 7
            self.data.twist.twist.linear.y,
            self.data.twist.twist.linear.z,
            self.data.twist.twist.angular.x,    # 10
            self.data.twist.twist.angular.y,
            self.data.twist.twist.angular.z
            ]

        return np.asarray(position_heading)

# def

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import visuals.records as R
    from mpl_toolkits import mplot3d



    topic_listener = RigTopicListener()
    topic_listener.listen_start()
    _ys = []
    for i in range(1000):
        _ys.append(topic_listener.get_sensory_observation())
        print(_ys[-1])
        sleep(0.01)
    _ys = np.asarray(_ys)

    rec = {
        "pos": _ys[:, :3],
        "head": _ys[:, 3:7],
        "vel_lin": _ys[:, 7:10],
        "vel_ang": _ys[:, 10:13],
    }

    R.save_records("data.hdf5", rec)
    R.print_record_shapes(rec)

    r = R.load_records("data.hdf5")[0]
    # ##
    # plt.title("X-Y")
    # plt.scatter(r["pos"][:, 0], r["pos"][:, 1], label="pos")
    # plt.legend()
    # ##
    # plt.figure()
    # plt.title("X-Z")
    # plt.scatter(r["pos"][:, 0], r["pos"][:, 2], label="pos")
    # ##

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(r["pos"][:, 0], r["pos"][:, 1], r["pos"][:, 2])

    ##
    fig = plt.figure()
    _f = fig.subplots(3, 3)
    for j, seg in enumerate(["pos", "vel_lin", "vel_ang"]):
        f = _f[j]
        for i, dim in enumerate(["x", "y", "z"]):
            f[i].plot(r[seg][:, i])
            if j == 1:
                f[i].set_xlabel(dim)
        f[0].set_ylabel(seg)


    plt.show()

