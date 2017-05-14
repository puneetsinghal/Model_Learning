cmake_minimum_required(VERSION 2.8.3)
project(model_learning)

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
find_package(Eigen3 REQUIRED)

SET (CMAKE_CXX_STANDARD 11)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (SONAME "hebi")
SET (MAJOR_VERSION "0") # C MAJOR VERSION
SET (MINOR_VERSION "12") # C MINOR VERSION
SET (CPP_VERSION "0") # CPP VERSION
SET (PKG_DIR ${CMAKE_BINARY_DIR}/${MAJOR_VERSION}.${MINOR_VERSION}.${CPP_VERSION})

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  CommandML.msg
  FeedbackML.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  trajectory_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime

)

include_directories(${catkin_INCLUDE_DIRS})

include_directories (
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src      # CPP wrapper classes
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include  # Underlying C headers
  ${Eigen_INCLUDE_DIRS}
  ) 

link_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  )
  
#add_executable(arm_openloop
#  src/arm_openloop.cpp
#  src/mac_address.cpp
#  src/command.cpp
#  src/feedback.cpp
#  src/info.cpp
#  src/group_command.cpp
#  src/group_feedback.cpp
#  src/group_info.cpp
#  src/module.cpp
#  src/group.cpp
#  src/lookup.cpp
#  src/arm_dynamics.cpp
#)
#target_link_libraries(arm_openloop ${catkin_LIBRARIES})
#add_dependencies(arm_openloop ${catkin_EXPORTED_TARGETS} model_learning_generate_messages_cpp)


add_executable(arm_controller
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
target_link_libraries(arm_controller ${catkin_LIBRARIES})
add_dependencies(arm_controller ${catkin_EXPORTED_TARGETS} model_learning_generate_messages_cpp)


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

#target_link_libraries(arm_openloop hebi m pthread)
target_link_libraries(arm_controller hebi m pthread)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
