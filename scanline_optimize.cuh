#pragma once
#include "ad_util.cuh"
#include "common_util.cuh"
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif


__global__ void scanline_optimize_left2right(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr,ADCensus_Option option);



__global__ void scanline_optimize_right2left(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr, ADCensus_Option option);



__global__ void scanline_optimize_top2bottom(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr, ADCensus_Option option);



__global__ void scanline_optimize_bottom2top(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr, ADCensus_Option option);





