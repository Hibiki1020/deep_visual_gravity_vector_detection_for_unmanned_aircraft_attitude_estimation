cmake_minimum_required(VERSION 3.0.2)
project(deep_visual_gravity_vector_detection_for_unmanned_aircraft_attitude_estimation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES deep_visual_gravity_vector_detection_for_unmanned_aircraft_attitude_estimation
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
