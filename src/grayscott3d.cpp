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
  const int32_t maxsteps = 200;

  // prepare a set of RNGs
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  // several range policies
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrpolicy({{0, 0, 0}}, {{n, n, n}});	// faster for openmp
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrpolicy({{0, 0, 0}}, {{n, n, n}}, {{64, 4, 2}});	// faster for cuda
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrpolicy({{0, 0, 0}}, {{n, n, n}}, {{2, 4, 64}});	// worse for both

  // begin timing
  Kokkos::Timer timer;
  timer.reset();

  // create the views
  Kokkos::View<ST***> u("U", n, n, n);
  Kokkos::View<ST***> v("V", n, n, n);
  Kokkos::View<ST***> dudt("dU/dt", n, n, n);
  Kokkos::View<ST***> dvdt("dV/dt", n, n, n);
  const double ncells = (double)n*n*n;

  // initialize the data
  Kokkos::parallel_for("init U",
                       mdrpolicy,
                       KOKKOS_LAMBDA (const int i, const int j, const int k) { u(i,j,k) = ui; } );
  Kokkos::parallel_for("init V",
                       mdrpolicy,
                       KOKKOS_LAMBDA (const int i, const int j, const int k) { v(i,j,k) = vi; } );

  Kokkos::fence();
  double time_init = timer.seconds();
  printf("time to init %e s\n", time_init);
  double time_laplace = 0.;
  double time_advect = 0.;

  // loop over time
  for (int step=0; step<maxsteps+1; ++step) {

  timer.reset();

  Kokkos::parallel_for("calc laplace U",
    mdrpolicy,
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
    mdrpolicy,
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
  Kokkos::fence();
  time_laplace += timer.seconds();

  timer.reset();
  Kokkos::parallel_for("finish d/dt",
    mdrpolicy,
    KOKKOS_LAMBDA (const int i, const int j, const int k) {
      const ST uv2 = u(i,j,k)*v(i,j,k)*v(i,j,k);
      auto generator = random_pool.get_state();
      dudt(i,j,k) += F*(1.0-u(i,j,k)) - uv2 + rs*(ST)generator.drand(-1., 1.);
      random_pool.free_state(generator);
      dvdt(i,j,k) += uv2 - (F+kr)*v(i,j,k);
    }
  );

  Kokkos::parallel_for("finish step",
    mdrpolicy,
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
  time_advect += timer.seconds();
  printf("."); fflush(stdout);

  // save as a png
  const int stepperout = 100;
  if (step%stepperout == 0) {
    char outfile[255];
    std::vector<unsigned char> imgdata(n*n);
    unsigned int error;
    int kslice = n/2;
    Kokkos::View<ST***>::HostMirror vh = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(vh, v);
    Kokkos::fence();

    //sprintf(outfile, "u_%04d.png", step/stepperout);
    //for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = u(i,j)*255;
    //error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
    //if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));

    sprintf(outfile, "v_%04d.png", step/stepperout);
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = vh(i,j,kslice)*255;
    error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
    if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));
  }

  // write interval time
  const int timerout = 10;
  if (step%timerout == 0) {
    static int last_step = 0;
    static double last_time = 0.;
    int step_interval = step - last_step + 1;
    double time_interval = (time_laplace+time_advect) - last_time;
    last_step = step+1;
    last_time = time_laplace+time_advect;
    printf("%d steps averaged %e s at %e GF/s\n", step_interval, time_interval/step_interval, step_interval*ncells*33./(time_interval*1e+9));
  }
  }

  double time_tot = time_laplace+time_advect;
  printf("  %d steps averaged %e s at %e GF/s\n", maxsteps+1, time_tot/(maxsteps+1), (maxsteps+1)*ncells*33./(time_tot*1e+9));
  printf("    laplace portion averaged %e s at %e GF/s\n", time_laplace/(maxsteps+1), (maxsteps+1)*ncells*16./(time_laplace*1e+9));
  printf("    update portion averaged %e s at %e GF/s\n", time_advect/(maxsteps+1), (maxsteps+1)*ncells*17./(time_advect*1e+9));

  // write all slices
  {
    char outfile[255];
    std::vector<unsigned char> imgdata(n*n);
    unsigned int error;
    Kokkos::View<ST***>::HostMirror vh = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(vh, v);
    Kokkos::fence();
    for (int k=0; k<n; ++k) {
      sprintf(outfile, "slice_%04d.png", k);
      for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) imgdata[i*n+j] = vh(i,j,k)*255;
      error = lodepng::encode(outfile, imgdata, n, n, LCT_GREY, 8);
      if (error) printf("encoder error %d: %s\n", error, lodepng_error_text(error));
    }
  }

  Kokkos::finalize();
}
