project(HELIADES)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(HELIADES_HEADERS
    heliades/common/random.h
    heliades/core/camera.h
    heliades/core/film.h
    heliades/core/hitable.h
    heliades/core/material.h
    heliades/geometry/ray.h
    heliades/geometry/sphere.h
)

set(HELIADES_SOURCES
    heliades/core/film.cu
)

enable_language("CUDA")
add_executable(main main.cu io.cu ${HELIADES_HEADERS} ${HELIADES_SOURCES})
# add_dependencies(main ponos)
target_include_directories(main PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
    ${PONOS_INCLUDE_DIR} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(main PRIVATE ${PONOS_LIBRARIES})
set_target_properties(main PROPERTIES
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_STANDARD 11
)
