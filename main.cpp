// mpic++ -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math main.cpp -o solverMPI
// mpirun -np 4 ./solverMPI 1D benchmarkX 250 100 0.0 1.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <mpi.h>

namespace program_options {

struct Options {
  unsigned int mpi_mode;  
  std::string name;
  size_t N;
  size_t iters;
  double fix_west;
  double fix_east;  
  void print() const {
    std::printf("mpi_mode: %u\nD", mpi_mode);    
    std::printf("name: %s\n", name.c_str());
    std::printf("N: %zu\n", N);
    std::printf("iters: %zu\n", iters);
    std::printf("fix_west: %lf\n", fix_west);
    std::printf("fix_east: %lf\n", fix_east);    
  }
};

auto parse(int argc, char *argv[]) {
  if (argc != 7)
    throw std::runtime_error("unexpected number of arguments");
  Options opts;
  if (std::string(argv[1]) == std::string("1D"))
    opts.mpi_mode = 1;
  else if( std::string(argv[1]) == std::string("2D"))
    opts.mpi_mode = 2;
  else
   throw std::runtime_error("invalid parameter for mpi_mode (valid are '1D' and '2D')");
  opts.name = argv[2];
  if (std::sscanf(argv[3], "%zu", &opts.N) != 1 && opts.N >= 2)
    throw std::runtime_error("invalid parameter for N");
  if (std::sscanf(argv[4], "%zu", &opts.iters) != 1 && opts.iters != 0)
    throw std::runtime_error("invalid parameter for iters");
  if (std::sscanf(argv[5], "%lf", &opts.fix_west) != 1)
    throw std::runtime_error("invalid value for fix_west");
  if (std::sscanf(argv[6], "%lf", &opts.fix_east) != 1)
    throw std::runtime_error("invalid value for fix_east");  
  return opts;
}

} // namespace program_options

int main(int argc, char *argv[]) try {

  // MPI initialization
  MPI_Init(&argc, &argv);

  // Rank and Size of processes
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure there are enough command-line arguments
  if (argc != 7) {
      if (rank == 0) { // Only the root process should print the error message
          std::cerr << "Usage: " << argv[0] << " <mpi_mode> <benchmark> <N> <iters> <fix_west> <fix_east>" << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1); // Abort if the arguments are incorrect
  }

  // parse args
  auto opts = program_options::parse(argc, argv);
  opts.print();
  std::cout << "-------------------------------------" << std::endl;

  // number of rows each process will handle
  size_t local_N = opts.N / size; // Basic number of rows per process
  size_t start_row = rank * local_N;
  size_t end_row = start_row + local_N - 1;
  // The last process gets the remainder if N is not divisible by the number of processes
  if (rank == size - 1) {
      end_row = opts.N - 1; // Last process extends to the end of the domain
  }
  
  // Allocate local grid including the ghost layers
  size_t local_height = end_row - start_row + 1; // +1 because end_row is inclusive
  size_t ghost_layers = 2;

  if (rank == 0 || rank == size - 1) {
      ghost_layers = 1; // Only one ghost layer for the first and last process
  }
  
  // local grid using a 1D vector
  std::vector<double> local_grid((local_height + ghost_layers) * opts.N, 0.0);


  // printing for debugging purposes
    for (int r = 0; r < size; r++) {
        if (rank == r) {
            // This rank's turn to print. Print also local size and ghost layers
            std::cout << "Rank: " << rank << " with local height: " << local_height << " and ghost layers: " << ghost_layers << std::endl;
            for (size_t i = 0; i < local_height + ghost_layers; ++i) {
                for (size_t j = 0; j < opts.N; ++j) {
                    std::cout << local_grid[i * opts.N + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "-------------------------------------" << std::endl;
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks wait here before the next rank starts printing
    }
  

  // initial guess (0.0) with fixed values in west (-100) and east (100)
  auto init = [N = opts.N, W = opts.fix_west, E = opts.fix_east]() -> auto {
    std::vector<double> res(N * N);
    for (size_t j = 0; j < N; ++j)
      for (size_t i = 0; i < N; ++i) {
        res[i + j * N] = 0.0;
        if (i % N == 0)
          res[i + j * N] = W;
        if (i % N == N - 1)
          res[i + j * N] = E;
      }
    return res;
  };

  // solver update
  auto jacobi_iter = [N = opts.N](const auto &xold, auto &xnew,
                                  bool residual = false) {
    auto h = 1.0 / (N - 1);
    auto h2 = h * h;
    // all interior points
    for (size_t j = 1; j < N - 1; ++j) {
      for (size_t i = 1; i < N - 1; ++i) {
        auto w = xold[(i - 1) + (j)*N];
        auto e = xold[(i + 1) + (j)*N];
        auto n = xold[(i) + (j + 1) * N];
        auto s = xold[(i) + (j - 1) * N];
        auto c = xold[(i) + (j)*N];
        if (!residual)
          xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
        else
          xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
      }
    }
    // isolating south boundary
    {
      size_t j = 0;
      for (size_t i = 1; i < N - 1; ++i) {
        auto w = xold[(i - 1) + (j)*N];
        auto e = xold[(i + 1) + (j)*N];
        auto n = xold[(i) + (j + 1) * N];
        auto s = n;
        auto c = xold[(i) + (j)*N];
        if (!residual)
          xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
        else
          xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
      }
    }
    // isolating north boundary
    {
      size_t j = N - 1;
      for (size_t i = 1; i < N - 1; ++i) {
        auto w = xold[(i - 1) + (j)*N];
        auto e = xold[(i + 1) + (j)*N];
        auto s = xold[(i) + (j - 1) * N];
        auto n = s;
        auto c = xold[(i) + (j)*N];
        if (!residual)
          xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
        else
          xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
      }
    }
  };

  // write vector to csv
  auto write = [N = opts.N, name = opts.name](const auto &x) -> auto {
    std::ofstream csv;
    csv.open(name + ".csv");
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i < N - 1; ++i) {
        csv << x[i + j * N] << " ";
      }
      csv << x[(N - 1) + j * N];
      csv << "\n";
    }
    csv.close();
  };

  // 2 norm
  auto norm2 = [N = opts.N](const auto &vec) -> auto {
    double sum = 0.0;
    for (size_t j = 0; j < N; ++j)
      for (size_t i = 1; i < (N - 1); ++i)
        sum += vec[i + j * N] * vec[i + j * N];

    return std::sqrt(sum);
  };

  // Inf norm
  auto normInf = [N = opts.N](const auto &vec) -> auto {
    double max = 0.0;
    for (size_t j = 0; j < N; ++j)
      for (size_t i = 1; i < (N - 1); ++i)
        max = std::fabs(vec[i + j * N]) > max ? std::fabs(vec[i + j * N]) : max;
    return max;
  };

  auto x1 = init();
  auto x2 = x1;
  for (size_t iter = 0; iter <= opts.iters; ++iter) {
    jacobi_iter(x1, x2);
    std::swap(x1, x2);
  }

  // write(b);

  write(x2);
  jacobi_iter(x1, x2, true);

  std::cout << "  norm2 = " << norm2(x2) << std::endl;
  std::cout << "normInf = " << normInf(x2) << std::endl;

  // MPI finalization
  MPI_Finalize();

  return EXIT_SUCCESS;
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}
