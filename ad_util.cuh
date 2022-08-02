#pragma once
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "common_util.cuh"
#include <math.h>
#include <inttypes.h>
#include <cstdlib>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_UCHAR3(pointer) (reinterpret_cast<uchar3*>(&(pointer))[0])
#define INVALID_FLOAT (INFINITY)



__device__ uint8_t hamming_distance(uint64_t val1, uint64_t val2);

///return census_transform value
///do boundary check inside ,return boundary value 0
__device__ uint64_t cencus_transform_9_7_d(uint8_t* img,int x, int y,int height,int width);//abandonded

//sub kernel for first step, cost init,compute census transform value, with hard code block size 32
__global__ void census_transform_97(uint8_t* gray_img, uint64_t* census_array, int height, int width);//finish
//sub kernel for first step, cost init,compute gray scale value
__global__ void compute_gray(uint8_t* color_img, uint8_t* gray_img,int height,int width);//finish

//main kernel for first step, cost init
__global__ void compute_cost(uint8_t* color_left,
                             uint8_t* gray_left,
                             uint8_t* color_right,
                             uint8_t* gray_right,
                             float* cost_init,
                             uint64_t* census_left,
                             uint64_t* census_right,
                             ADCensus_Option option,
                             int height,
                             int width
                             );//finish







