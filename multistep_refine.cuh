#pragma once
#include "ad_util.cuh"
#include "common_util.cuh"
#include <math.h>
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif

#define OCCLUSIONS 1
#define MISMATCHES 2
#define NORMAL 0

//hxwxd
__global__ void retrieve_left_disp(float* cost, float* disp_left, ADCensus_Option option);


__global__ void retrieve_right_disp(float* cost, float* disp_left, ADCensus_Option option);


//LRCheck
__global__ void outlier_detection(float* disp_left, float* disp_right, float* disp_out, uint8_t* disp_mask, ADCensus_Option option);


/*
  region voting with one iteration, implementation are slow
*/
__global__ void region_voting(float* disp_left, float* disp_out, uint8_t* disp_mask, CrossArm* arms, ADCensus_Option option);

//8 direction aggregation
__global__ void interpolation(float* disp_left, float* disp_out,uint8_t* img_left, uint8_t* disp_mask, ADCensus_Option option);



__global__ void median_filter(float* disp_left, float* disp_out, int height, int width);

