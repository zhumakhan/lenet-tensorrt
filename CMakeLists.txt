cmake_minimum_required(VERSION 2.8) # Проверка версии CMake.
									# Если версия установленой программы
									# старее указаной, произайдёт аварийный выход.

add_executable(main network.cpp)		# Создает исполняемый файл с именем main
									# из исходника main.cpp

set_source_files_properties(network.cpp PROPERTIES LANGUAGE CUDA)

# add_library(nvinfer SHARED /usr/local/lib/libnvinfer.so)
# set_target_properties(nvinfer PROPERTIES LINKER_LANGUAGE CUDA)
# set(CMAKE_NVCC_CREATE_SHARED_LIBRARY )

add_library(nvinfer SHARED IMPORTED)
set_property(TARGET nvinfer PROPERTY IMPORTED_LOCATION "/usr/local/lib/libnvinfer.so")
set(CMAKE_CUDA_LINK_EXECUTABLE true)
set(CMAKE_CUDA_COMPILE_OBJECT true)
set(_CMAKE_CUDA_WHOLE_FLAG true)