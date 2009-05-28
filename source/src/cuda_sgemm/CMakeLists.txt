##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# The include_directories adds paths to both C/C++ compilation in the
# native compiler and cuda compilation in NVCC.
INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_CUT_INCLUDE_DIR}
  )

##############################################################################
# There are four ways to compile source files with NVCC. 

CUDA_ADD_EXECUTABLE(cuda_som
  cuda.cu
  )

TARGET_LINK_LIBRARIES(cuda_som
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_CUT_LIBRARY}
  )

# Or compile the cuda code into a shared library.
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE_old ${CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE})
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE true)
#CUDA_ADD_LIBRARY(test_lib
#  test_lib.cu
#  external_dependency.h
#  )
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ${CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE_old})

# Then link the shared library to another executable.
#ADD_EXECUTABLE(lib_example
#  main.cc
#  )

# Specify the dependency.
#TARGET_LINK_LIBRARIES(lib_example
#  test_lib
#  )

# Using the CUDA_COMPILE macro
#CUDA_COMPILE(CUDA_FILES test_bin.cu)

#ADD_EXECUTABLE(cuda_compile_example
#  ${CUDA_FILES}
#  main.cc
#  external_dependency.h
#  )
#TARGET_LINK_LIBRARIES(cuda_compile_example
# ${CUDA_LIBRARIES}
# )

# Generating PTX files. 
# CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)

# Add a special target to clean nvcc generated files.
#CUDA_BUILD_CLEAN_TARGET()


