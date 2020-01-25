add_executable(hello_cuda hello_cuda.cu)
add_executable(fft fft.cu)
# add_dependencies(fft ponos)
target_include_directories(fft PUBLIC 
    ${PONOS_INCLUDE_DIR} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(fft PRIVATE ${PONOS_LIBRARIES} cufft)
set_target_properties(fft PROPERTIES
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_STANDARD 11
)

add_subdirectory(heliades)
add_subdirectory(optix_rt)