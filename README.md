# grayscott
Run a two-parameter Gray-Scott reaction-diffusion simulation using Kokkos for performance portability

![textureimage](media/v_0010.png?raw=true "2D simulation after 1000 steps")

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
	./src/grayscott3d.bin 256

The first command will run a 2D simulation and write out files that are 4096x4096 pixels.
The second will run a 3D simulation of a 256x256x256 cube, writing a center plane every
100 steps, and at the end will write each plane as a png file. These sizes were selected
because each runs the simulation over 2^24 cells. This symmetry is also achievable with
32768 and 1024, respectively (1B cells, 16 GB memory needed).

## Performance
On a 16-core AMD Ryzen 9 3950X, the 4096^2 problem solves at ~0.24s per step,
and a 256^3 solves in ~0.075s per step.
On a 3070Ti with Cuda 11.2, the 4096^2 problem achieves 0.005s and the 256^3 requires 0.008s.

## Background

## Thanks
[Lodepng](https://github.com/lvandeve/lodepng), [Kokkos](https://github.com/kokkos/kokkos)
