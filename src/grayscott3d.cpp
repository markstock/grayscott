//
// grayscott3d.cpp
//
// Run a 3D, two-component, Gray-Scott reaction diffusion simulation
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
  printf("size of fields %ld x %ld x %ld\n", n, n, n);

  using ST = float;

  const ST ui = 0.5;		// initial uniform value for U
  const ST vi = 0.5;
  const ST Du = 0.4;		// diffusion rate of U
  const ST Dv = 0.2;
  const ST F  = 0.02;		// growth rate of U vs. V
  const ST kr = 0.048;		// additional kill rate of V
  const ST rs = 0.1;		// scale of random noise
  const ST dt = 1.0;
  const int32_t maxsteps = 2000;

  // prepare a set of RNGs
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  // several range policies
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> policyA({{0, 0, 0}}, {{n, n, n}});

  // begin timing
  Kokkos::Timer timer;
  timer.reset();

  // create the views
  Kokkos::View<ST***> u("U", n, n, n);
  Kokkos::View<ST***> v("V", n, n, n);
  Kokkos::View<ST***> dudt("dU/dt", n, n, n);
  Kokkos::View<ST***> dvdt("dV/dt", n, n, n);

  // initialize the data
  Kokkos::parallel_for("init U",
                       policyA,
                       KOKKOS_LAMBDA (const int i, const int j, const int k) { u(i,j,k) = ui; } );
  Kokkos::parallel_for("init V",
                       policyA,
                       KOKKOS_LAMBDA (const int i, const int j, const int k) { v(i,j,k) = vi; } );

  Kokkos::fence();
  double time_64 = timer.seconds();
  printf("time to init %e s\n", time_64);

  // loop over time
  for (int step=0; step<maxsteps+1; ++step) {

  timer.reset();

  Kokkos::parallel_for("calc laplace U",
    policyA,
    KOKKOS_LAMBDA (const int i, const int j, const int k) {
      const int im1 = (i==0 ? n-1 : i-1);
      const int ip1 = (i==n-1 ? 0 : i+1);
      const int jm1 = (j==0 ? n-1 : j-1);
      const int jp1 = (j==n-1 ? 0 : j+1);
      const int km1 = (k==0 ? n-1 : k-1);
      const int kp1 = (k==n-1 ? 0 : k+1);
      dudt(i,j,k) = Du * ((u(im1,j,k)+u(ip1,j,k)+u(i,jm1,k)+u(i,jp1,k)+u(i,j,km1)+u(i,j,kp1))/6. - u(i,j,k));
    }
  );

  Kokkos::parallel_for("calc laplace V",
    policyA,
    KOKKOS_LAMBDA (const int i, const int j, const int k) {
      const int im1 = (i==0 ? n-1 : i-1);
      const int ip1 = (i==n-1 ? 0 : i+1);
      const int jm1 = (j==0 ? n-1 : j-1);
      const int jp1 = (j==n-1 ? 0 : j+1);
      const int km1 = (k==0 ? n-1 : k-1);
      const int kp1 = (k==n-1 ? 0 : k+1);
      dvdt(i,j,k) = Dv * ((v(im1,j,k)+v(ip1,j,k)+v(i,jm1,k)+v(i,jp1,k)+v(i,j,km1)+v(i,j,kp1))/6. - v(i,j,k));
    }
  );

  Kokkos::parallel_for("finish d/dt",
    policyA,
    KOKKOS_LAMBDA (const int i, const int j, const int k) {
      const ST uv2 = u(i,j,k)*v(i,j,k)*v(i,j,k);
      auto generator = random_pool.get_state();
      dudt(i,j,k) += F*(1.0-u(i,j,k)) - uv2 + rs*(ST)generator.drand(-1., 1.);
      random_pool.free_state(generator);
      dvdt(i,j,k) += uv2 - (F+kr)*v(i,j,k);
    }
  );

  Kokkos::parallel_for("finish step",
    policyA,
    KOKKOS_LAMBDA (const int i, const int j, const int k) {
      u(i,j,k) += dt * dudt(i,j,k);
      v(i,j,k) += dt * dvdt(i,j,k);
      if (u(i,j,k) > 1.0) u(i,j,k) = 1.0;
      if (u(i,j,k) < 0.0) u(i,j,k) = 0.0;
      if (v(i,j,k) > 1.0) v(i,j,k) = 1.0;
      if (v(i,j,k) < 0.0) v(i,j,k) = 0.0;
    }
  );

  Kokkos::fence();
  time_64 = timer.seconds();
  printf("  step %d took %e s at %e GF/s\n", step, time_64, n*n*(double)26*n/(time_64*1e+9));

  // save as a png
  if (step%10 == 0) {
    char outfile[255];
    std::vector<unsigned char> imgdata(n*n);
    unsigned int error;
    int kslice = n/2;

    //sprintf(outfile, "u_%04d.png", step/10);
    //for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = u(i,j)*255;
    //error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
    //if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));

    sprintf(outfile, "v_%04d.png", step/10);
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = v(i,j,kslice)*255;
    error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
    if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));
  }
  }

  // write all slices
  {
    char outfile[255];
    std::vector<unsigned char> imgdata(n*n);
    unsigned int error;
    for (int k=0; k<n; ++k) {
      sprintf(outfile, "slice_%04d.png", k);
      for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = v(i,j,k)*255;
      error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
      if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));
    }
  }

  Kokkos::finalize();
}
