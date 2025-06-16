// saxpy.cu (after adding event timing)

#include <cuda_runtime.h>
#include <iostream>

__global__
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

float saxpyAndTime(int N, float a, float *h_x, float *h_y) {
    float *d_x, *d_y;

    size_t size = N * sizeof(float);
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMemcpy(d_x, h_x, size,cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, size,cudaMemcpyHostToDevice);

    // Timing with CUDA events
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    saxpy<<<blocks, threads>>>(N, a, d_x, d_y);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

    float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);

    // Cleanup
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaMemcpy(h_y, d_y, size,cudaMemcpyDeviceToHost);
   cudaFree(d_y);
   cudaFree(d_x);

    return milliseconds;
}

int main()
{
    float a = 2.0f;

    for (int exp = 15; exp <= 25; ++exp) {
        int N = 1 << exp;

        float *h_x = new float[N];
        float *h_y = new float[N];
        for (int i = 0; i < N; ++i) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }

        float milliseconds = saxpyAndTime(N, a, h_x, h_y);
        std::cout << N << "," << milliseconds << "ms" <<std::endl;

        delete[] h_y;
        delete[] h_x;
    }

    return 0;
}
