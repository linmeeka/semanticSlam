cmake_minimum_required(VERSION 2.8)
project(DynaSLAM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
  # SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O0 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(OpenCV_DIR "/home/kylin/libs/opencv-2.4.13.6/build/installed/share/OpenCV")

message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV 2.4.11 QUIET)
if(NOT OpenCV_FOUND)
    message("OpenCV > 2.4.11 not found.")
    find_package(OpenCV 3.0 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 3.0 not found.")
    endif()
endif()

#find_package(OpenCV 3.0 QUIET)
#if(NOT OpenCV_FOUND)
#   find_package(OpenCV 2.4.3 QUIET)
#   if(NOT OpenCV_FOUND)
#      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#   endif()
#endif()

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Test REQUIRED)

find_package(PythonLibs REQUIRED)
if (NOT PythonLibs_FOUND)
    message(FATAL_ERROR "PYTHON LIBS not found.")
else()
    message("PYTHON LIBS were found!")
    message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
endif()

find_package(Boost REQUIRED COMPONENTS thread)
if(Boost_FOUND)
    message("Boost was found!")
    message("Boost Headers DIRECTORY: " ${Boost_INCLUDE_DIRS})
    message("Boost LIBS DIRECTORY: " ${Boost_LIBRARY_DIRS})
    message("Found Libraries: " ${Boost_LIBRARIES})
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(PCL 1.8 REQUIRED)
find_package(OpenMP)
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
/usr/include/python2.7/
/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
${Boost_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/gco-v3.0
)

add_definitions(${PCL_DEFINITIONS} )

set(HEADERS
    include/gco-v3.0/graph.h
    include/gco-v3.0/block.h
    include/gco-v3.0/energy.h
    include/gco-v3.0/GCoptimization.h
    include/gco-v3.0/LinkedBlockList.h
    include/config.h
    include/lsa_tr.h
    include/segmentation.h
    include/segmentation_helpers.h
    )

message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Conversion.cc
src/MaskNet.cc
src/Geometry.cc
src/pointcloudmapping.cc
src/segmentation.cc
src/SegData.cc
sec/Model.cc
gco-v3.0/maxflow.cpp gco-v3.0/maxflow.cpp gco-v3.0/LinkedBlockList.cpp gco-v3.0/GCoptimization.cpp gco-v3.0/graph.cpp ${HEADERS}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
/usr/lib/x86_64-linux-gnu/libpython2.7.so
${Boost_LIBRARIES}
${PCL_LIBRARIES}
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

#add_executable(mono_carla
#Examples/Monocular/mono_carla.cc)
#target_link_libraries(mono_carla ${PROJECT_NAME})

#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz/associations_r_d.txt /home/kylin/git/DynaSLAM/walking_rgbd  /home/kylin/git/DynaSLAM/output

#./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml /home/kylin/dataset/rgbd_dataset_freiburg1_desk /home/kylin/git/DynaSLAM/mask_mono
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg1_desk /home/kylin/dataset/TUM/rgbd_dataset_freiburg1_desk/association.txt /home/kylin/git/DynaSLAM/mask_desk /home/kylin/git/DynaSLAM/mask_desk_output
# ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz /home/kylin/git/DynaSLAM/mask
#./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_halfsphere /home/kylin/git/DynaSLAM/mask_sit_mono
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_halfsphere /home/kylin/git/DynaSLAM/Examples/RGB-D/associations/fr3_sitting_halfsphere.txt /home/kylin/git/DynaSLAM/mask_sit_rgbd /home/kylin/git/DynaSLAM/output_sit_rgbd

#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_xyz/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/walking_xyz  /home/kylin/git/DynaSLAM/masks/walking_xyz/output
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_xyz /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_xyz/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/sitting_xyz  /home/kylin/git/DynaSLAM/masks/sitting_xyz/output
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_static /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_static/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/walking_static  /home/kylin/git/DynaSLAM/masks/walking_static/output
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_static /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_static/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/sitting_static  /home/kylin/git/DynaSLAM/masks/sitting_static/output
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_rpy /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_walking_rpy/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/walking_rpy  /home/kylin/git/DynaSLAM/masks/walking_rpy/output
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_rpy /home/kylin/dataset/TUM/rgbd_dataset_freiburg3_sitting_rpy/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/sitting_rpy  /home/kylin/git/DynaSLAM/masks/sitting_rpy/output
#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM2.yaml /home/kylin/dataset/TUM/rgbd_dataset_freiburg2_desk_with_person /home/kylin/dataset/TUM/rgbd_dataset_freiburg2_desk_with_person/associations_r_d.txt /home/kylin/git/DynaSLAM/masks/desk_with_person  /home/kylin/git/DynaSLAM/masks/desk_with_person/output