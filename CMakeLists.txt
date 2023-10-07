cmake_minimum_required(VERSION 3.16)
project(grayscott2d CXX)

add_subdirectory(src)
add_subdirectory(kokkos)

#add_executable(grayscott2d.bin grayscott2d.cpp)

#target_link_libraries(grayscott2d.bin Kokkos::kokkos)

