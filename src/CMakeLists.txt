
include_directories(.)

add_executable(grayscott2d.bin grayscott2d.cpp lodepng.cpp)
target_link_libraries(grayscott2d.bin Kokkos::kokkos)

add_executable(grayscott3d.bin grayscott3d.cpp lodepng.cpp)
target_link_libraries(grayscott3d.bin Kokkos::kokkos)

