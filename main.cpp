#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// -------------------- deterministic pattern (exact correctness) --------------------
static inline uint32_t mix_u32(uint32_t x) {
  // Simple integer mixing (deterministic)
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static inline int encode_value(int src, int dst, int k) {
  // Unique-ish value per (src, dst, k). Fits in signed 32-bit int.
  uint32_t x = 0;
  x ^= mix_u32(static_cast<uint32_t>(src + 1) * 1000003U);
  x ^= mix_u32(static_cast<uint32_t>(dst + 1) * 916191U);
  x ^= mix_u32(static_cast<uint32_t>(k + 1) * 972663749U);
  // keep it positive-ish
  return static_cast<int>(x & 0x7fffffffU);
}

// -------------------- TODO: implement this --------------------
void MPI_Alltoall_int(const int* sb, int* rb, int msg_size, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (int i = 0; i < size; ++i) {
        // Calculate the partner to exchange data with in this step
        // This pattern helps distribute network traffic
        int target = i; 

        // Pointer to the block of data intended for 'target'
        const int* send_ptr = sb + (target * msg_size);
        
        // Pointer to the block where data from 'target' should be stored
        int* recv_ptr = rb + (target * msg_size);

        if (rank == target) {
            // Local copy for self-to-self communication
            std::copy(send_ptr, send_ptr + msg_size, recv_ptr);
        } else {
            // Use Sendrecv to avoid manual deadlock management
            MPI_Sendrecv(send_ptr, msg_size, MPI_INT, target, 0,
                         recv_ptr, msg_size, MPI_INT, target, 0,
                         comm, MPI_STATUS_IGNORE);
        }
    }
}

// -------------------- helpers --------------------
static std::vector<int> parse_sizes_arg(const std::string& s) {
  // expects "1,128,4096"
  std::vector<int> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) out.push_back(std::stoi(token));
  }
  return out;
}

static void fill_sendbuf(std::vector<int>& sb, int msg_size, int rank, int size) {
  // sb has size*msg_size. Block dst is intended for dst.
  for (int dst = 0; dst < size; ++dst) {
    for (int k = 0; k < msg_size; ++k) {
      sb[dst * msg_size + k] = encode_value(rank, dst, k);
    }
  }
}

static long long count_errors(const std::vector<int>& rb, int msg_size, int rank, int size) {
  long long errors = 0;
  for (int src = 0; src < size; ++src) {
    for (int k = 0; k < msg_size; ++k) {
      int got = rb[src * msg_size + k];
      int exp = encode_value(src, rank, k);
      if (got != exp) errors++;
    }
  }
  return errors;
}

static double time_ms(double t0, double t1) { return (t1 - t0) * 1000.0; }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Defaults
  int iters = 20;
  std::vector<int> sizes = {1, 128, 4096, 100000, 500000};

  // Parse args
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else if (a == "--sizes" && i + 1 < argc) {
      sizes = parse_sizes_arg(argv[++i]);
    }
  }

  if (size < 2) {
    if (rank == 0) std::cerr << "Need at least 2 MPI processes.\n";
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    std::cout << "MPI HW3: All-to-All via point-to-point\n";
    std::cout << "Processes: " << size << "\n";
    std::cout << "Iters per size: " << iters << "\n";
    std::cout << "Sizes (ints): ";
    for (size_t i = 0; i < sizes.size(); ++i) {
      std::cout << sizes[i] << (i + 1 < sizes.size() ? ", " : "\n");
    }
    std::cout << "---------------------------------------------------\n";
  }

  bool global_pass = true;

  for (int msg_size : sizes) {
    std::vector<int> sb(size * msg_size);
    std::vector<int> rb_ref(size * msg_size);
    std::vector<int> rb_custom(size * msg_size);

    fill_sendbuf(sb, msg_size, rank, size);

    // Warm-up barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------- baseline: MPI_Alltoall --------------------
    double t0_ref = MPI_Wtime();
    for (int it = 0; it < iters; ++it) {
      MPI_Alltoall(sb.data(), msg_size, MPI_INT, rb_ref.data(), msg_size, MPI_INT, MPI_COMM_WORLD);
    }
    double t1_ref = MPI_Wtime();

    long long local_err_ref = count_errors(rb_ref, msg_size, rank, size);
    long long total_err_ref = 0;
    MPI_Reduce(&local_err_ref, &total_err_ref, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // -------------------- custom: student implementation --------------------
    MPI_Barrier(MPI_COMM_WORLD);

    double t0_c = MPI_Wtime();
    for (int it = 0; it < iters; ++it) {
      // Clear buffer)
      std::fill(rb_custom.begin(), rb_custom.end(), -1);
      MPI_Alltoall_int(sb.data(), rb_custom.data(), msg_size, MPI_COMM_WORLD);
    }
    double t1_c = MPI_Wtime();

    long long local_err_c = count_errors(rb_custom, msg_size, rank, size);
    long long total_err_c = 0;
    MPI_Reduce(&local_err_c, &total_err_c, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      double ref_ms = time_ms(t0_ref, t1_ref) / iters;
      double cus_ms = time_ms(t0_c, t1_c) / iters;
      std::cout << "msg_size=" << msg_size
                << " | MPI_Alltoall: " << ref_ms << " ms"
                << " | Custom: " << cus_ms << " ms"
                << " | ratio=" << (cus_ms / std::max(ref_ms, 1e-12)) << "\n";
      std::cout << "    errors: baseline=" << total_err_ref
                << ", custom=" << total_err_c << "\n";
      std::cout << "---------------------------------------------------\n";
    }

    bool pass = true;
    if (rank == 0) {
      if (total_err_ref != 0) {
        std::cerr << "[WARNING] Baseline MPI_Alltoall errors != 0 (unexpected)\n";
      }
      if (total_err_c != 0) pass = false;
    }

    // Broadcast pass/fail to everyone
    int pass_i = pass ? 1 : 0;
    MPI_Bcast(&pass_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    global_pass = global_pass && (pass_i == 1);
  }

  if (rank == 0) {
    std::cout << (global_pass ? "RESULT: PASS\n" : "RESULT: FAIL\n");
  }

  MPI_Finalize();
  return global_pass ? 0 : 2;
}
