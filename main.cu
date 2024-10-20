#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda.h>

#define FILE_PATH "primes.txt"
#define THREADS_PER_BLOCK 256
#define SEGMENT_SIZE (1 << 22)

__device__ inline long long cuda_max(long long a, long long b) {
    return (a > b) ? a : b;
}

__global__ void sieve_kernel(char* d_is_prime, long long low, long long high, int* d_primes, int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_primes; i += blockDim.x * gridDim.x) {
        int p = d_primes[i];
        long long p_long = (long long)p;
        long long start = cuda_max(p_long * p_long, ((low + p_long - 1) / p_long) * p_long);
        if (start > high) continue;
        for (long long j = start; j <= high; j += p_long) {
            d_is_prime[j - low] = 0;
        }
    }
}

std::vector<int> simple_sieve(int limit) {
    int sqrt_limit = static_cast<int>(std::sqrt(limit)) + 1;
    std::vector<char> is_prime(sqrt_limit + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    for (int p = 3; p * p <= sqrt_limit; p += 2) {
        if (is_prime[p]) {
            for (int i = p * p; i <= sqrt_limit; i += 2 * p) {
                is_prime[i] = 0;
            }
        }
    }
    std::vector<int> primes = {2};
    for (int p = 3; p <= sqrt_limit; p += 2) {
        if (is_prime[p]) {
            primes.push_back(p);
        }
    }
    return primes;
}

long long get_last_prime() {
    std::ifstream infile(FILE_PATH, std::ios::in | std::ios::ate);
    if (!infile.is_open()) {
        return 1;
    }
    std::streampos size = infile.tellg();
    if (size == 0) {
        infile.close();
        return 1;
    }
    char ch;
    infile.seekg(-1, std::ios_base::end);
    while (infile.tellg() > 0) {
        infile.get(ch);
        if (ch == '\n') {
            break;
        }
        infile.seekg(-2, std::ios_base::cur);
    }
    std::string last_line;
    getline(infile, last_line);
    infile.close();
    if (!last_line.empty()) {
        return std::stoll(last_line);
    }
    return 1;
}

void bulk_save_primes(const std::vector<long long>& primes) {
    std::ofstream outfile(FILE_PATH, std::ios::app);
    for (const auto& prime : primes) {
        outfile << prime << "\n";
    }
    outfile.close();
}

int main() {
    long long current = get_last_prime() + 1;
    long long total_primes = 0;
    int max_digits = 0;
    int batch_counter = 0;
    bool running = true;

    auto start_time = std::chrono::steady_clock::now();

    while (running) {
        auto batch_start_time = std::chrono::steady_clock::now();

        long long high = current + SEGMENT_SIZE - 1;
        int sqrt_high = static_cast<int>(std::sqrt(high)) + 1;
        std::vector<int> primes = simple_sieve(sqrt_high);

        char* d_is_prime;
        int* d_primes;
        int num_primes = primes.size();

        cudaMalloc((void**)&d_is_prime, SEGMENT_SIZE * sizeof(char));
        cudaMalloc((void**)&d_primes, num_primes * sizeof(int));

        cudaMemcpy(d_primes, primes.data(), num_primes * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemset(d_is_prime, 1, SEGMENT_SIZE * sizeof(char));

        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocks = (num_primes + threadsPerBlock - 1) / threadsPerBlock;

        sieve_kernel<<<blocks, threadsPerBlock>>>(d_is_prime, current, high, d_primes, num_primes);
        cudaDeviceSynchronize();

        std::vector<char> h_is_prime(SEGMENT_SIZE);
        cudaMemcpy(h_is_prime.data(), d_is_prime, SEGMENT_SIZE * sizeof(char), cudaMemcpyDeviceToHost);

        std::vector<long long> segment_primes;
        for (long long i = 0; i < SEGMENT_SIZE; ++i) {
            if (h_is_prime[i]) {
                segment_primes.push_back(current + i);
            }
        }

        long long primes_found_in_batch = segment_primes.size();
        total_primes += primes_found_in_batch;

        if (!segment_primes.empty()) {
            long long longest_prime = segment_primes.back();
            int digits = std::to_string(longest_prime).length();
            if (digits > max_digits) {
                max_digits = digits;
            }
            bulk_save_primes(segment_primes);
        }

        cudaFree(d_is_prime);
        cudaFree(d_primes);

        batch_counter++;
        auto batch_end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> batch_runtime = batch_end_time - batch_start_time;

        std::cout << "Batch " << batch_counter << ": Found " << primes_found_in_batch
                  << " primes. | Total Primes: " << total_primes
                  << " | Longest Prime Digits: " << max_digits
                  << " | Batch Runtime: " << batch_runtime.count() << " seconds" << std::endl;

        current = high + 1;
    }

    auto total_runtime = std::chrono::steady_clock::now() - start_time;
    std::cout << "Total Runtime: " << std::chrono::duration_cast<std::chrono::seconds>(total_runtime).count()
              << " seconds" << std::endl;

    return 0;
}