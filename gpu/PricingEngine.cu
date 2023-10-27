#include "PricingEngine.h"

#include <cmath>
#include <climits>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <iostream>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;
#include <curand_kernel.h>

using std::string;


__device__ double d_lowPrice[2][100000] = {0.0};
__device__ double d_finalPrice[2][100000] = {0.0};


// RNG init kernel
__global__ void initRNG(curandState *const rngStates, const unsigned int seed, int numSims) {
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = tid; i < numSims; i += step) {
        curand_init(seed + i, 0, 0, &rngStates[i]);
        curand_init(seed + i + numSims, 0, 0, &rngStates[i + numSims]);
    }
}



__device__ inline void getPathStep(double &drift1, double &diffusion1, double &drift2, double &diffusion2,
 const double &corr, curandState &state1, curandState &state2, double *ans) {
    double g1 = curand_normal_double(&state1);
    double g2 = curand_normal_double(&state2);


    double x1 = g1;
    double x2 = corr * g1 + std::sqrt(1 - corr * corr) * g2;

    
    ans[0] = std::exp(drift1 + diffusion1 * x1);
    ans[1] = std::exp(drift2 + diffusion2 * x2);

}


__device__ inline double discount(double r, double maturity){
    return exp(-r*maturity);
}

__device__ int binary_search(const double *lv, double target, int start, int end) {
    int l = start;
    int r = end;

    while (l<=r) {
        int mid = l + (r-l)/2;
        if (lv[mid] == target) {
            return mid;
        } else if (lv[mid] < target) {
            l = mid + 1;
        } else r = mid - 1;
    }
    return r>0?r:0;
}



__global__ void generatePath(curandState *const rngStates, const struct Market* market, const int numSims) {

    const double maturity = market->maturity;

    const double strike1 = market->strike[0];
    const double strike2 = market->strike[1];

    const double r = market->r;

    const int num_dts = market->num_dts;
    const int num_lv1 = market->num_lv1;
    const int num_lv2 = market->num_lv2;


    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;


    curandState localState1;
    curandState localState2;

    double *lowPriceOutput1;
    double *lowPriceOutput2;
    double *finalPriceOutput1;
    double *finalPriceOutput2;

    double spot1;
    double spot2;
    double lowp1;
    double lowp2;

    double drift1;
    double diffusion1;
    int index1;

    double drift2;
    double diffusion2;
    int index2;

    double elapsed;


    for (unsigned int i = tid; i < numSims; i += step) {
        
        localState1 = rngStates[i];
        localState2 = rngStates[i + numSims];

        lowPriceOutput1 =  d_lowPrice[0] + i;
        lowPriceOutput2 =  d_lowPrice[1] + i;
        finalPriceOutput1 =  d_finalPrice[0] + i;
        finalPriceOutput2 =  d_finalPrice[1] + i;

        spot1 = market->spots[0];
        spot2 = market->spots[1];
        lowp1 = spot1;
        lowp2 = spot2;


        elapsed = 0.0;


        for (int j = 0; j < num_dts; j++) {
            index1 = binary_search(market->lv_index1, (maturity - elapsed)/(log(spot1/strike1) + 0.001), 0, num_lv1-1);
            index2 = binary_search(market->lv_index2, (maturity - elapsed)/(log(spot2/strike2) + 0.001), 0, num_lv2-1);


            drift1 = (r - static_cast<double>(0.5) * market->lv1[j][index1] * market->lv1[j][index1]) * market->dts[j];

            diffusion1 = market->lv1[j][index1] * std::sqrt(market->dts[j]);

            drift2 = (r - static_cast<double>(0.5) * market->lv2[j][index2] * market->lv2[j][index2]) * market->dts[j];

            diffusion2 = market->lv2[j][index2] * std::sqrt(market->dts[j]);

            double exp_value[2] = {0.0};
            getPathStep(drift1, diffusion1, drift2, diffusion2, market->corr, localState1, localState2, exp_value);

            spot1 *= exp_value[0];
            spot2 *= exp_value[1];

            lowp1 = lowp1 > spot1?spot1:lowp1;
            lowp2 = lowp2 > spot2?spot2:lowp2;

            elapsed += market->dts[j];

            // printf("spot1: %lf, spot2: %lf\n", spot1, spot2);

        }


        *lowPriceOutput1 = lowp1;
        *lowPriceOutput2 = lowp2;

        *finalPriceOutput1 = spot1;
        *finalPriceOutput2 = spot2;

    }

    
}

__device__ void computeValue(struct Option *option, int numSims) {
    double value = 0.0;
    for (unsigned int i = 0; i < numSims; i++) {
        if (option->barrier1 > d_lowPrice[0][i] || option->barrier2 > d_lowPrice[1][i]) continue;
        else {
            double p1 = (d_finalPrice[0][i] - option->strike1 > 0.0)?(d_finalPrice[0][i] - option->strike1):0.0;
            double p2 = (d_finalPrice[1][i] - option->strike2 > 0.0)?(d_finalPrice[1][i] - option->strike2):0.0;
            value += p1 > p2?p2:p1;
        }                
    }
    value /= numSims;
    value *= discount(option->r, option->maturity);
    option->value = value;
    //printf("The value is %.2lf", value);
}

__global__ void cudaComputePortValue(struct Portfolio *port, int size, int numSims) {
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    for (int i = tid; i < size; i += step) {
        //printf("%d", i);
        computeValue(&port->option_list[i], numSims);
    }
}




pricingEngine::pricingEngine(unsigned int seed, unsigned int numSims, unsigned int threadBlockSize, unsigned int device): 
        m_seed(seed), m_numSims(numSims), m_threadBlockSize(threadBlockSize), m_device(device) {}






void pricingEngine::simulate(const struct Market* market) {
    cudaError_t cudaResult = cudaSuccess;
    struct cudaDeviceProp deviceProperties;
    struct cudaFuncAttributes funcAttributes;

    // Get device properties
    cudaResult = cudaGetDeviceProperties(&deviceProperties, m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get device properties: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Check precision is valid
    unsigned int deviceVersion =
        deviceProperties.major * 10 + deviceProperties.minor;

    if (deviceVersion < 13) {
        throw std::runtime_error("Device does not have double precision support");
    }

    // Attach to GPU
    cudaResult = cudaSetDevice(m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not set CUDA device: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    block.x = m_threadBlockSize;
    grid.x = (m_numSims + m_threadBlockSize - 1) / m_threadBlockSize;

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 10;
    unsigned int numSMs = deviceProperties.multiProcessorCount;

    while (grid.x > 2 * blocksPerSM * numSMs) {
        grid.x >>= 1;
    }

    //std::cout<<"test"<<std::endl;

    cudaResult = cudaFuncGetAttributes(&funcAttributes, initRNG);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get function attributes: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
        throw std::runtime_error(
            "Block X dimension is too large for initRNG kernel");
    }

    //std::cout<<"test"<<std::endl;
    // Get generatePaths function properties and check the maximum block size
    cudaResult = cudaFuncGetAttributes(&funcAttributes, generatePath);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get function attributes: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
        throw std::runtime_error(
            "Block X dimension is too large for generatePath kernel");
    }

    // std::cout<<"test"<<std::endl;


    struct Market *d_market = 0;
    cudaResult = cudaMalloc((void **)&d_market, sizeof(struct Market));

    if (cudaResult != cudaSuccess) {
        string msg("Could not allocate memory on device for market data: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    cudaResult = cudaMemcpy(d_market, market, sizeof(struct Market), cudaMemcpyHostToDevice);

    if (cudaResult != cudaSuccess) {
        string msg("Could not copy data to device: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }



    curandState *d_rngStates = 0;
    cudaResult = cudaMalloc((void **)&d_rngStates, 2 * m_numSims * sizeof(curandState));

    if (cudaResult != cudaSuccess) {
        string msg("Could not allocate memory on device for RNG state: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // std::cout<<"test"<<std::endl;

    initRNG<<<grid, block>>>(d_rngStates, m_seed, m_numSims);
    cudaDeviceSynchronize();
    // Generate paths
    generatePath<<<grid, block>>>(d_rngStates, d_market, m_numSims);
    cudaDeviceSynchronize();

    // Cleanup
    if (d_market) {
        cudaFree(d_market);
        d_market = 0;
    }

    if (d_rngStates) {
        cudaFree(d_rngStates);
        d_rngStates = 0;
    }


}




void pricingEngine::computePortValue(struct Portfolio *port) const {
    cudaError_t cudaResult = cudaSuccess;
    struct cudaDeviceProp deviceProperties;
    struct cudaFuncAttributes funcAttributes;

    // Get device properties
    cudaResult = cudaGetDeviceProperties(&deviceProperties, m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get device properties: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Check precision is valid
    unsigned int deviceVersion =
        deviceProperties.major * 10 + deviceProperties.minor;

    if (deviceVersion < 13) {
        throw std::runtime_error("Device does not have double precision support");
    }

    // Attach to GPU
    cudaResult = cudaSetDevice(m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not set CUDA device: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Determine how to divide the work between cores
    int size = port->size;
    dim3 block;
    dim3 grid;
    block.x = m_threadBlockSize;
    grid.x = (size + m_threadBlockSize - 1) / m_threadBlockSize;

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 10;
    unsigned int numSMs = deviceProperties.multiProcessorCount;

    while (grid.x > 2 * blocksPerSM * numSMs) {
        grid.x >>= 1;
    }

    cudaResult = cudaFuncGetAttributes(&funcAttributes, cudaComputePortValue);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get function attributes: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
        throw std::runtime_error(
            "Block X dimension is too large for cudaComputePortValue");
    }


    double value = 0.0;


    struct Portfolio *d_port = 0;
    cudaResult = cudaMalloc((void **)&d_port, sizeof(struct Portfolio));

    if (cudaResult != cudaSuccess) {
        string msg("Could not allocate memory on device for Portfolio: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    cudaResult = cudaMemcpy(d_port, port, sizeof(struct Portfolio), cudaMemcpyHostToDevice);

    if (cudaResult != cudaSuccess) {
        string msg("Could not copy data to device: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }




    cudaComputePortValue<<<grid, block>>>(d_port, size, m_numSims);
    cudaDeviceSynchronize();

    cudaResult = cudaMemcpy(port, d_port, sizeof(struct Portfolio),
                                cudaMemcpyDeviceToHost);



    if (cudaResult != cudaSuccess) {
        string msg("Could not copy results to host: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

        

    if (d_port) {
        cudaFree(d_port);
        d_port = 0;
    }


    for (int i = 0; i < size; i++) {
        value += port->option_list[i].value;
    }

    port->value = value;
}

