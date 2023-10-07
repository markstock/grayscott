//
// grayscott2d.cpp
//
// Run a 2D, two-component, Gray-Scott reaction diffusion simulation
//

#include "lodepng.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <vector>
#include <cstdio>


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [<kokkos_options>] <array size>\n", argv[0]);
    Kokkos::finalize();
    exit(1);
  }

  const int n = std::stoi(argv[1]);
  printf("size of fields %ld x %ld\n", n, n);

  using ST = float;

  const ST ui = 0.5;		// initial uniform value for U
  const ST vi = 0.5;
  const ST Du = 0.8;		// diffusion rate of U
  const ST Dv = 0.4;
  const ST F  = 0.02;	// growth rate of U vs. V
  const ST k  = 0.048;	// additional kill rate of V
  const ST rs = 0.1;		// scale of random noise
  const ST dt = 1.0;
  const int32_t maxsteps = 200;

  // prepare a set of RNGs
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  // several range policies
  // this works well for CPU (except doesn't do SIMD)
  //Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrpolicy({{0, 0}}, {{n, n}});
  // this works for CUDA
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrpolicy({{0, 0}}, {{n, n}}, {{128,4}});

  // begin timing
  Kokkos::Timer timer;
  timer.reset();

  // create the views
  Kokkos::View<ST**> u("U", n, n);
  Kokkos::View<ST**> v("V", n, n);
  Kokkos::View<ST**> dudt("dU/dt", n, n);
  Kokkos::View<ST**> dvdt("dV/dt", n, n);

  // initialize the data
  Kokkos::parallel_for("init U",
                       mdrpolicy,
                       KOKKOS_LAMBDA (const int i, const int j) { u(i,j) = ui; } );
  Kokkos::parallel_for("init V",
                       mdrpolicy,
                       KOKKOS_LAMBDA (const int i, const int j) { v(i,j) = vi; } );

  Kokkos::fence();
  double time_64 = timer.seconds();
  printf("time to init %e s\n", time_64);

  // loop over time
  for (int step=0; step<maxsteps+1; ++step) {

  timer.reset();

  Kokkos::parallel_for("calc laplace U",
    mdrpolicy,
    KOKKOS_LAMBDA (const int i, const int j) {
      const int im1 = (i==0 ? n-1 : i-1);
      const int ip1 = (i==n-1 ? 0 : i+1);
      const int jm1 = (j==0 ? n-1 : j-1);
      const int jp1 = (j==n-1 ? 0 : j+1);
      dudt(i,j) = Du * (0.25*(u(im1,j)+u(ip1,j)+u(i,jm1)+u(i,jp1)) - u(i,j));
    }
  );

  Kokkos::parallel_for("calc laplace V",
    mdrpolicy,
    KOKKOS_LAMBDA (const int i, const int j) {
      const int im1 = (i==0 ? n-1 : i-1);
      const int ip1 = (i==n-1 ? 0 : i+1);
      const int jm1 = (j==0 ? n-1 : j-1);
      const int jp1 = (j==n-1 ? 0 : j+1);
      dvdt(i,j) = Dv * (0.25*(v(im1,j)+v(ip1,j)+v(i,jm1)+v(i,jp1)) - v(i,j));
    }
  );

  Kokkos::parallel_for("finish d/dt",
    mdrpolicy,
    KOKKOS_LAMBDA (const int i, const int j) {
      const ST uv2 = u(i,j)*v(i,j)*v(i,j);
      auto generator = random_pool.get_state();
      dudt(i,j) += F*(1.0-u(i,j)) - uv2 + rs*(ST)generator.drand(-1., 1.);
      random_pool.free_state(generator);
      dvdt(i,j) += uv2 - (F+k)*v(i,j);
    }
  );

  Kokkos::parallel_for("finish step",
    mdrpolicy,
    KOKKOS_LAMBDA (const int i, const int j) {
      u(i,j) += dt * dudt(i,j);
      v(i,j) += dt * dvdt(i,j);
      if (u(i,j) > 1.0) u(i,j) = 1.0;
      if (u(i,j) < 0.0) u(i,j) = 0.0;
      if (v(i,j) > 1.0) v(i,j) = 1.0;
      if (v(i,j) < 0.0) v(i,j) = 0.0;
    }
  );

  Kokkos::fence();
  time_64 = timer.seconds();
  printf("  step %d took %e s at %e GF/s\n", step, time_64, n*n*(double)26/(time_64*1e+9));

  // save as a png
  const int stepperout = 100;
  if (step%stepperout == 0) {
    char outfile[255];
    std::vector<unsigned char> imgdata(n*n);
    unsigned int error;
    Kokkos::View<ST**>::HostMirror vh = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(vh, v);
    Kokkos::fence();

    //sprintf(outfile, "u_%04d.png", step/stepperout);
    //for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = u(i,j)*255;
    //error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
    //if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));

    sprintf(outfile, "v_%04d.png", step/stepperout);
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = vh(i,j)*255;
    error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
    if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));
  }
  }

  Kokkos::finalize();
}
