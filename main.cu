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

__global__ void sieve_kernel(char* d_is_prime, long long low, long long high, int* d_primes, int num_primes) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long num = low + idx; num <= high; num += stride) {
        if (num % 2 == 0 && num > 2) {
            d_is_prime[num - low] = 0;
            continue;
        }
        
        char is_prime = 1;
        for (int j = 0; j < num_primes; ++j) {
            int p = d_primes[j];
            if (p * p > num) break;
            if (num % p == 0) {
                is_prime = 0;
                break;
            }
        }
        d_is_prime[num - low] = is_prime;
    }
}

std::vector<int> simple_sieve(int limit) {
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (int p = 2; p * p <= limit; ++p) {
        if (is_prime[p]) {
            for (int i = p * p; i <= limit; i += p) {
                is_prime[i] = false;
            }
        }
    }

    std::vector<int> primes;
    for (int p = 2; p <= limit; ++p) {
        if (is_prime[p]) {
            primes.push_back(p);
        }
    }
    return primes;
}

long long get_last_prime() {
    std::ifstream infile(FILE_PATH, std::ios::in);
    if (!infile.is_open()) {
        return 1;
    }
    infile.seekg(-1, std::ios_base::end);
    if (infile.peek() == '\n') {
        infile.seekg(-1, std::ios_base::cur);
        int i = infile.tellg();
        for (; i > 0; i--) {
            infile.seekg(i, std::ios_base::beg);
            if (infile.peek() == '\n') {
                infile.get();
                break;
            }
        }
    } else {
        infile.seekg(0, std::ios_base::beg);
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

        int blocks = (SEGMENT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        sieve_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_is_prime, current, high, d_primes, num_primes);

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