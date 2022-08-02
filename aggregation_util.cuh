#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include "common_util.cuh"
#include "ad_util.cuh"
#include <cmath>
#include <cooperative_groups.h>
#define MAX_ARM_LENGTH 255

using namespace cooperative_groups;
namespace cg = cooperative_groups;



//this function could be optimize using 4 seperate kernel and share memory
/*
__global__ void BuildArmLeft(uint8_t* color_img_left, CrossArm* cross_arms, ADCensus_Option option);
__global__ void BuildArmRight(uint8_t* color_img_left, CrossArm* cross_arms, ADCensus_Option option);
__global__ void BuildArmTop(uint8_t* color_img_left, CrossArm* cross_arms, ADCensus_Option option);
__global__ void BuildArmBottom(uint8_t* color_img_left, CrossArm* cross_arms, ADCensus_Option option);
*/

//
//build arm in 2d grid style
//original is build arm in 1dx1d grid style,but 1d block is too big for large image so we give up
__global__ void BuildArm(uint8_t* color_img_left,CrossArm* cross_arms,ADCensus_Option option);

//inner function for build arm
__device__ void FindHorizontalArm(uint8_t* color_img_left,int width, int height, uint8_t& left, uint8_t& right, ADCensus_Option option);

//inner function for build arm
__device__ void FindVerticalArm(uint8_t* color_img_left, int width, int height, uint8_t& top, uint8_t& bottom, ADCensus_Option option);

//in case you gpu do not support cooperative group
__global__ void ComputeSubpixelCountHorizontal(CrossArm* cross_arms,uint16_t* count_buffer, ADCensus_Option option);
//2nd step, counting the subpixel count,i dont use cooperative group here but using cooperative group could be much easy
__global__ void ComputeSubpixelAggregateHorizontal(CrossArm* cross_arms, uint16_t* horizontal_count, uint16_t* count_buffer, ADCensus_Option option);


//this is the specific implmentation
__global__ void ComputeSubpixelCountHorizontalCooperativeGroup(CrossArm* cross_arms,uint16_t* horizontal_count, uint16_t* count_buffer, ADCensus_Option option);

__global__ void ComputeSubpixelCountVertical(CrossArm* cross_arms, uint16_t* count_buffer, ADCensus_Option option);

__global__ void ComputeSubpixelAggregateVertical(CrossArm* cross_arms, uint16_t* vertical_count, uint16_t* count_buffer, ADCensus_Option option);

__global__ void ComputeSubpixelCountVerticalCooperativeGroup(CrossArm* cross_arms, uint16_t* vertical_count, uint16_t* count_buffer, ADCensus_Option option);

//multi stream/multiple block implementation of aggregation,launch in a per disp per kernel way with share memory loading
//multi stream or multiple blocks could also ok
//use cost buffer(3D Volume to buffer (actuallly may be cost_aggr) to buffer the cost aggregation value in certain disparity value
//
__global__ void AggregateInArms1stphase(float* cost_init, float* cost_buffer,CrossArm* cross_arms,bool horizontal_first, ADCensus_Option option);

__global__ void AggregateInArms2ndphase(float* cost_init, float* cost_buffer, CrossArm* cross_arms, uint16_t* count, bool horizontal_first, ADCensus_Option option);





