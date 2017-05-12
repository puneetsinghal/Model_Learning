cmake_minimum_required (VERSION 2.6)
project (model_learning)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  trajectory_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

SET (CMAKE_CXX_STANDARD 11)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (SONAME "hebi")
SET (MAJOR_VERSION "0") # C MAJOR VERSION
SET (MINOR_VERSION "12") # C MINOR VERSION
SET (CPP_VERSION "0") # CPP VERSION
SET (PKG_DIR ${CMAKE_BINARY_DIR}/${MAJOR_VERSION}.${MINOR_VERSION}.${CPP_VERSION})


## Start Richard Added


################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  CommandML.msg
  FeedbackML.msg
)

# Generate services in the 'srv' folder
#add_service_files(
# FILES
#)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  trajectory_msgs
  sensor_msgs
)



###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime
  DEPENDS system_lib
)



###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
## End Richard Added

include_directories (
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src      # CPP wrapper classes
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include  # Underlying C headers
  ${Eigen_INCLUDE_DIRS}
  ) 

link_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  )
  
add_executable(arm_openloop
# hebi_listener 
#   src/hebi_listener.cpp
  src/arm_openloop.cpp
  src/mac_address.cpp
  src/command.cpp
  src/feedback.cpp
  src/info.cpp
  src/group_command.cpp
  src/group_feedback.cpp
  src/group_info.cpp
  src/module.cpp
  src/group.cpp
  src/lookup.cpp
  src/arm_dynamics.cpp
)
TARGET_LINK_LIBRARIES(arm_openloop ${catkin_LIBRARIES})
ADD_DEPENDENCIES(arm_openloop ${catkin_EXPORTED_TARGETS})

add_executable(arm_controller
# hebi_listener 
#   src/hebi_listener.cpp
  src/arm_controller.cpp
  src/mac_address.cpp
  src/command.cpp
  src/feedback.cpp
  src/info.cpp
  src/group_command.cpp
  src/group_feedback.cpp
  src/group_info.cpp
  src/module.cpp
  src/group.cpp
  src/lookup.cpp
  src/arm_dynamics.cpp
)
TARGET_LINK_LIBRARIES(arm_controller ${catkin_LIBRARIES})
ADD_DEPENDENCIES(arm_controller ${catkin_EXPORTED_TARGETS})

# These are all of the source files required to build the C++ wrapper for the
# C API.
add_library(hebi_cpp_files OBJECT
  src/mac_address.cpp
  src/command.cpp
  src/feedback.cpp
  src/info.cpp
  src/group_command.cpp
  src/group_feedback.cpp
  src/group_info.cpp
  src/module.cpp
  src/group.cpp
  src/lookup.cpp
)

# target_link_libraries(hebi_listener ${catkin_LIBRARIES} hebi m pthread )
target_link_libraries(arm_openloop hebi m pthread)
target_link_libraries(arm_controller hebi m pthread)
