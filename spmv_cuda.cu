// spmv_cuda.cu — CS406 HW2 CUDA Skeleton
//
// Build (example):
//   nvcc -O3 -std=c++17 spmv_cuda.cu -o spmv_cuda
// Run:
//   ./spmv_cuda path/to/matrix.mtx
//
// You will implement:
//   - device_build_from_global_csr()
//   - spmv_cuda_kernel(...) (may be a controller or call other kernels)
//   - spmv(...): any launches/transfers you need
//
// We provide: CPU MatrixMarket → CSR loader into global CPU arrays.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(err)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} } while(0)

// ---------------- CPU-side global CSR ----------------
int G_N = 0;                   // rows
int G_M = 0;                   // cols
std::vector<int>    G_row_ptr; // length: G_N+1
std::vector<int>    G_col_idx; // length: nnz
std::vector<double> G_vals;    // length: nnz

double gflops = 0.0; // set by your implementation if you choose

// ---------------- Device pointers (you will populate) ----------------
static int    *d_row_ptr = nullptr;
static int    *d_col_idx = nullptr;
static double *d_vals    = nullptr;
static double *d_x       = nullptr;
static double *d_y = nullptr;

// ---------------- MatrixMarket → CSR loader (CPU) ----------------
static void read_matrix_market_to_global_csr(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open file: " + path);

    std::string header;
    if (!std::getline(fin, header)) throw std::runtime_error("Empty file");
    if (header.rfind("%%MatrixMarket", 0) != 0)
        throw std::runtime_error("Not a MatrixMarket file");

    const bool is_coordinate = (header.find("coordinate") != std::string::npos);
    const bool is_array      = (header.find("array")      != std::string::npos);
    const bool is_pattern    = (header.find("pattern")    != std::string::npos);
    const bool is_symmetric  = (header.find("symmetric")  != std::string::npos);
    const bool is_general    = (header.find("general")    != std::string::npos);

    if (!is_coordinate && !is_array)
        throw std::runtime_error("Only 'coordinate' or 'array' MatrixMarket supported");

    std::string line;
    do {
        if (!std::getline(fin, line)) throw std::runtime_error("Missing size line");
    } while (!line.empty() && line[0] == '%');

    long nrows=0, ncols=0, nz_or_vals=0;
    {
        std::istringstream ss(line);
        if (is_coordinate) {
            if (!(ss >> nrows >> ncols >> nz_or_vals))
                throw std::runtime_error("Bad size line for coordinate");
        } else {
            if (!(ss >> nrows >> ncols))
                throw std::runtime_error("Bad size line for array");
        }
    }

    struct Trip { int r, c; double v; };
    std::vector<Trip> trips;

    if (is_coordinate) {
        if (!is_general && !is_symmetric)
            throw std::runtime_error("Unsupported symmetry flag in coordinate header");
        trips.reserve(is_symmetric ? (size_t)nz_or_vals * 2 : (size_t)nz_or_vals);

        for (long t = 0; t < nz_or_vals; ++t) {
            if (!std::getline(fin, line)) throw std::runtime_error("Unexpected EOF");
            if (line.empty()) { --t; continue; }
            std::istringstream s(line);
            int i, j; double v = 1.0;
            if (!(s >> i >> j)) throw std::runtime_error("Bad coordinate entry");
            if (!is_pattern) { if (!(s >> v)) v = 1.0; }
            --i; --j;
            if (i < 0 || j < 0) continue;
            trips.push_back({i, j, v});
            if (is_symmetric && i != j) trips.push_back({j, i, v});
        }

        G_N = (int)nrows; G_M = (int)ncols;
        std::vector<std::vector<std::pair<int,double>>> rows(G_N);
        for (auto &t : trips) {
            if (t.r >= 0 && t.r < G_N && t.c >= 0 && t.c < G_M)
                rows[t.r].push_back({t.c, t.v});
        }
        G_row_ptr.assign(G_N + 1, 0);
        for (int r = 0; r < G_N; ++r) {
            auto &vec = rows[r];
            std::sort(vec.begin(), vec.end(), [](auto &a, auto &b){ return a.first < b.first; });
            int w = 0;
            for (int u = 0; u < (int)vec.size();) {
                int c = vec[u].first;
                double s = vec[u].second;
                int v = u + 1;
                while (v < (int)vec.size() && vec[v].first == c) { s += vec[v].second; ++v; }
                vec[w++] = {c, s};
                u = v;
            }
            vec.resize(w);
            G_row_ptr[r+1] = G_row_ptr[r] + (int)vec.size();
        }
        const int nnz = G_row_ptr.back();
        G_col_idx.resize(nnz); G_vals.resize(nnz);
        for (int r = 0; r < G_N; ++r) {
            int base = G_row_ptr[r];
            for (int k = 0; k < (int)rows[r].size(); ++k) {
                G_col_idx[base + k] = rows[r][k].first;
                G_vals   [base + k] = rows[r][k].second;
            }
        }
    } else {
        if (ncols != 2)
            throw std::runtime_error("array real general must be N x 2 (edge list)");
        const long N = nrows;
        std::vector<double> colmajor; colmajor.reserve(N * 2);

        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::istringstream s(line);
            double v;
            if (!(s >> v)) throw std::runtime_error("Bad array value line");
            colmajor.push_back(v);
        }
        if ((long)colmajor.size() != N * 2)
            throw std::runtime_error("Unexpected value count in array file");

        std::vector<int> U(N), V(N);
        for (long r = 0; r < N; ++r) {
            long iu = (long)std::llround(colmajor[(size_t)r + 0 * (size_t)N]);
            long iv = (long)std::llround(colmajor[(size_t)r + 1 * (size_t)N]);
            if (iu <= 0 || iv <= 0) continue;
            U[r] = (int)(iu - 1);
            V[r] = (int)(iv - 1);
        }

        int nmax = 0;
        for (long r = 0; r < N; ++r)
            nmax = std::max(nmax, std::max(U[r], V[r]));
        G_N = G_M = nmax + 1;

        std::vector<std::vector<int>> rows(G_N);
        for (long r = 0; r < N; ++r) {
            if (U[r] >= 0 && U[r] < G_N && V[r] >= 0 && V[r] < G_M)
                rows[U[r]].push_back(V[r]);
        }

        G_row_ptr.assign(G_N + 1, 0);
        for (int i = 0; i < G_N; ++i) {
            auto &vec = rows[i];
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
            G_row_ptr[i+1] = G_row_ptr[i] + (int)vec.size();
        }
        const int nnz = G_row_ptr.back();
        G_col_idx.resize(nnz); G_vals.assign(nnz, 1.0);
        for (int i = 0; i < G_N; ++i) {
            int base = G_row_ptr[i];
            for (int k = 0; k < (int)rows[i].size(); ++k)
                G_col_idx[base + k] = rows[i][k];
        }
    }
}

// Sorts rows by length (descending) and updates column indices 
// to maintain correctness (symmetry). This helps load balancing on GPU.
void preprocess_reorder_matrix() {
    std::cerr << "[Preprocessing] Performing Symmetric Reordering (Load Balance)...\n";
    
    //Calculate row lengths and store original indices
    std::vector<std::pair<int, int>> row_stats(G_N); 
    for (int i = 0; i < G_N; ++i) {
        row_stats[i] = {G_row_ptr[i+1] - G_row_ptr[i], i};
    }

    //Sort rows: Longest rows first
    
    std::sort(row_stats.begin(), row_stats.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; 
    });

    //Build the Permutation Map (Old Index -> New Index)
    std::vector<int> perm_old_to_new(G_N);
    for (int new_idx = 0; new_idx < G_N; ++new_idx) {
        int old_idx = row_stats[new_idx].second;
        perm_old_to_new[old_idx] = new_idx;
    }

    //Build reordered CSR arrays
    std::vector<int> new_row_ptr(G_N + 1, 0);
    std::vector<int> new_col_idx;
    std::vector<double> new_vals;
    
    new_col_idx.reserve(G_col_idx.size());
    new_vals.reserve(G_vals.size());

    for (int i = 0; i < G_N; ++i) {
        int old_row_idx = row_stats[i].second;
        int start = G_row_ptr[old_row_idx];
        int end = G_row_ptr[old_row_idx + 1];

        for (int k = start; k < end; ++k) {
            int old_col = G_col_idx[k];
            // Apply symmetric permutation to the column index
            new_col_idx.push_back(perm_old_to_new[old_col]);
            new_vals.push_back(G_vals[k]);
        }
        new_row_ptr[i+1] = new_col_idx.size();
    }

    //Replace Global Arrays
    G_row_ptr = new_row_ptr;
    G_col_idx = new_col_idx;
    G_vals = new_vals;
}

// --------------- You will build device state here -------------------
static void device_build_from_global_csr() {
    // You will allocate device memory and copy CSR (and any vectors you keep on device).
    size_t x_bytes       = G_M * sizeof(double); // CORRECTED from G_N
    size_t y_bytes       = G_N * sizeof(double);
    size_t row_ptr_bytes = (G_N + 1) * sizeof(int);
    size_t col_idx_bytes = G_col_idx.size() * sizeof(int);
    size_t vals_bytes    = G_vals.size() * sizeof(double);

    // 1. Allocate Device Memory
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, row_ptr_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_col_idx, col_idx_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_vals, vals_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_x, x_bytes)); // Use G_N for size of x/y
    CUDA_CHECK(cudaMalloc((void**)&d_y, y_bytes));

    // 2. Copy Matrix Data (Host -> Device)
    CUDA_CHECK(cudaMemcpy(d_row_ptr, G_row_ptr.data(), row_ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, G_col_idx.data(), col_idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,    G_vals.data(),    vals_bytes,    cudaMemcpyHostToDevice));
}

// --------------- You will implement your CUDA kernel(s) -------------
__global__ void spmv_cuda_kernel(int num_rows, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const double* __restrict__ vals, const double* __restrict__ x, double* __restrict__ y) {
    // You will implement your CUDA kernel(s).
    // Global Thread ID
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double sum = 0.0;
        int start = row_ptr[row];
        int end   = row_ptr[row + 1];

        // Loop over the non-zero elements of this row
        for (int k = start; k < end; ++k) {
            // A[row][col] * x[col]
            sum += vals[k] * x[col_idx[k]];
        }
        y[row] = sum;
    }
}

// --------------- Host wrapper that calls your kernel(s) -------------
// Includes event timing around YOUR launch(es).
static void spmv(const std::vector<double>& x_host, std::vector<double>& y_host) {
    // You will do any H2D/D2H copies you need.
    
    size_t x_bytes = G_M * sizeof(double); 
    size_t y_bytes = G_N * sizeof(double);

    CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), x_bytes, cudaMemcpyHostToDevice));

    int block_size = 128; 
    int grid_size  = (G_N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // YOU WILL LAUNCH YOUR KERNEL(S) HERE.
    // Example: spmv_cuda_kernel<<<grid_dim, block_dim>>>(...);
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        // Launch Kernel: y = A * x
        spmv_cuda_kernel<<<grid_size, block_size>>>(G_N, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
        
        // Swap pointers for next iteration
        // The output 'd_y' becomes the input 'd_x' for the next step
        std::swap(d_x, d_y);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    // We need the result back to verify correctness
    CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double sec = milliseconds * 1e-3;
    const size_t nnz = G_col_idx.size();
    if (sec > 0.0) {
        // One SpMV: ≈ 2*nnz FLOPs (mul+add)
        gflops = (2.0 * (double)nnz) / (sec * 1e9);
    }
    // You will ensure y_host is filled if you want a non-zero checksum.
}

// --------------- Cleanup (we do this) -------------------------------
static void device_free_all() {
    if (d_row_ptr) { 
        CUDA_CHECK(cudaFree(d_row_ptr)); 
        d_row_ptr = nullptr; 
    }
    if (d_col_idx) { 
        CUDA_CHECK(cudaFree(d_col_idx)); 
        d_col_idx = nullptr; 
    }
    if (d_vals) { 
        CUDA_CHECK(cudaFree(d_vals));    
        d_vals = nullptr; 
    }
    if (d_x) { 
        CUDA_CHECK(cudaFree(d_x));       
        d_x = nullptr; 
    }
    if (d_y) { 
        CUDA_CHECK(cudaFree(d_y));       
        d_y = nullptr; 
    }
}

// ------------------------------- main -------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./spmv_cuda path/to/matrix.mtx\n";
        return 1;
    }
    const std::string path = argv[1];

    std::cerr << "[load] " << path << "\n";
    read_matrix_market_to_global_csr(path);
    const size_t nnz = G_col_idx.size();
    std::cerr << "Matrix: n=" << G_N << " m=" << G_M << " nnz=" << nnz << "\n";

    if (G_N == 0 || G_M == 0 || nnz == 0) {
        std::cerr << "Empty/invalid matrix.\n";
        return 1;
    }

    preprocess_reorder_matrix();

    std::vector<double> x(G_M, 1.0), y(G_N, 0.0);

    device_build_from_global_csr();
    spmv(x, y);

    double chk = 0.0; for (double v : y) chk += v;
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "gflops=" << gflops
              << "  checksum=" << chk << "\n";

    device_free_all();
    return 0;
}
