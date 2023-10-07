# grayscott
Run a two-parameter Gray-Scott reaction-diffusion simulation using Kokkos for performance portability

## Build
The following works for me on my AMD Ryzen 3950X:

	git clone --recurse-submodules https://github.com/markstock/grayscott
	cd grayscott
	mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN2=ON ..
	make

And for CUDA:

	cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DKokkos_ENABLE_CUDA=ON ..

## Run
Both programs take one argument: the size of the array in each dimension.

	./src/grayscott2d.bin 4096
	./src/grayscott3d.bin 1024

The first command will run a 2D simulation and write out files that are 4096x4096 pixels.
The second will run a 3D simulation of a 1024x1024x1024 cube, writing a center plane every
100 steps, and at the end will write each plane as a png file.

## Performance
On a 16-core AMD Ryzen 9 3950X, a 2000^2 problem solves at ~0.02s per step, and a 256^3 solves in ~0.08s per step.
Adding block sizes to the execution policy allows 0.0015 sec per step for the 2048^2 solution, but that's still
only 70 GF/s on a 3070 Ti.

## Background

## Thanks
[Lodepng](https://github.com/lvandeve/lodepng), [Kokkos](https://github.com/kokkos/kokkos)
