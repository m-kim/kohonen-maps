

# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(SOM)
#LINK_DIRECTORIES(
#/home/mbk/GotoBLAS/
#)
#INCLUDE_DIRECTORIES(
#/home/mbk/GotoBLAS
#)
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
FIND_PACKAGE(CUDA)
FIND_PACKAGE(GLUT)
FIND_PACKAGE(OpenGL)
FIND_PACKAGE(Qt4 REQUIRED)

SUBDIRS(
  src
  )

 SET(CMAKE_BUILD_TYPE debug)
