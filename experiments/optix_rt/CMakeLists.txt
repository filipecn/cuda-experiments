# set(optix_LIBRARY "")
find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED VERSION 7.0)

find_program(BIN2C bin2c
        DOC "Path to the cuda-sdk bin2c executable.")
set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> -o <OBJECT> <FLAGS> -c <SOURCE>")
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -o <OBJECT> <FLAGS> -c <SOURCE>")
macro(cuda_compile_and_embed output_var cuda_file)
    set(c_var_name ${output_var})
    cuda_compile_ptx(ptx_files ${cuda_file})
    list(GET ptx_files 0 ptx_file)
    set(embedded_file ${ptx_file}_embedded.c)
    add_custom_command(
            OUTPUT ${embedded_file}
            COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
            DEPENDS ${ptx_file}
            COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
    set(${output_var} ${embedded_file})
endmacro()

include_directories(
        ${OptiX_INCLUDE}
        ${PONOS_INCLUDE_DIR}
)

cuda_compile_and_embed(embedded_ptx_code device_programs.cu)

set(headers
        optix_context.h
        optix_module.h
        optix_pipeline.h
        optix_program_group.h
        debug.h
        renderer.h
        )

set(sources
        optix_context.cpp
        optix_module.cpp
        optix_pipeline.cpp
        optix_program_group.cpp
        renderer.cpp
        )

add_executable(glass_example glass_example.cpp ${sources} ${headers} ${embedded_ptx_code})
# add_dependencies(glass_example ponos)
target_include_directories(glass_example PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PONOS_INCLUDE_DIR}
        ${OptiX_INCLUDE}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(glass_example PRIVATE
        ${PONOS_LIBRARIES}
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        )
