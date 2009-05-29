CUDA_ADD_EXECUTABLE(som_mt
  main.cc
  )

TARGET_LINK_LIBRARIES(som_mt
	goto_opteronp-r1.26
    
)