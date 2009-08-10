SET(QT_USE_QTOPENGL 1)
INCLUDE(${QT_USE_FILE})

QT_WRAP_CPP(cuda_som
MOC_FILES 
QtSOMWidget.h
window.h
)
message("ARGH ${MOC_FILES} ")
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# The include_directories adds paths to both C/C++ compilation in the
# native compiler and cuda compilation in NVCC.

INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_CUT_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIR}
  ${QT_INCLUDE_DIR}
  )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shared.h.in ${CMAKE_CURRENT_SOURCE_DIR}/shared.h)
IF(APPLE)
LINK_DIRECTORIES(
        ${CUDA_SDK_ROOT_DIR}/common/lib/darwin/
)
ENDIF()

CUDA_ADD_EXECUTABLE(cuda_som
	window.cc
	DensitySOMWidget.cc
	QtSOMWidget.cc
	HistSOMWidget.cc
	SOM.cc
	main.cc
  cuda.cu
  ${MOC_FILES}
  )

TARGET_LINK_LIBRARIES(cuda_som
	${CUDA_CUBLAS_LIBRARIES}
	${CUDA_CUT_LIBRARY}
	lapack
	${OPENGL_LIBRARY}
	${GLUT_LIBRARY}
	${QT_LIBRARIES}
	
)