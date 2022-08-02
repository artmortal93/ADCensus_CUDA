#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include "common_util.cuh"
#include <stdint.h>
#include <iostream>

class ADCensusStereo {
public:
	ADCensus_Option m_option;
	uint8_t* img_left_rgb_h = nullptr;
	uint8_t* img_right_rgb_h = nullptr;
	uint8_t* img_left_rgb_d = nullptr;
	uint8_t* img_right_rgb_d = nullptr;
	uint8_t* img_left_gray_d = nullptr;
	uint8_t* img_right_gray_d = nullptr;
	uint64_t* census_left_d = nullptr;
	uint64_t* census_right_d = nullptr;
	//when doing optimization, we bouncing the the boostrap optimization process using only two volume to speed up
	float* cost_init_d=nullptr; //3d aggr volume
	float* cost_aggr_d=nullptr; //3d aggr volume
	CrossArm* cross_arm_left_d = nullptr; //2d cross arm
	uint16_t* vec_counter_horizontal_d=nullptr;
	uint16_t* vec_counter_vertical_d = nullptr;
	uint16_t* vec_counter_buffer_d = nullptr;
	//for post processing
	uint8_t* disp_mask_d = nullptr;
	float* disp_left_buffer_d = nullptr;
	float* disp_left_d = nullptr;
	float* disp_right_d = nullptr;
	float* disp_left_h = nullptr; //host mem
	float* disp_right_h = nullptr;//host mem
	unsigned int width = 0;
	unsigned int height = 0;
	bool first_time = true;
	bool speedup_use_multiple_stream = true;                                            
	const static int stream_num = 2;
	cudaStream_t streams[stream_num];
	ADCensusStereo() :m_option() {};
	ADCensusStereo(ADCensus_Option option) :m_option(option) {
		width = m_option.width;
		height = m_option.height;
	};

	~ADCensusStereo();
public:
	//public interface
	void Init();
	void Reset();
	void SetOption(ADCensus_Option option) {
		m_option = option;
		width = m_option.width;
		height = m_option.height;
	}
	void SetComputeImg(uint8_t* left_img, uint8_t* right_img);
	void Compute();
	//return the pointer from itself
	float* RetrieveLeftDisparity();
    float* RetrieveRightDisparity();

protected:
	//clean up the memory of the all the buffer to proper state in order to reuse the cuda memory(for not first time usage)
	void CleanUpMemory();
	//allocate first time use memory on device
	void AllocateCudaResource(); 
	void FreeCudaResource();
	void CostCompute();
	void CostAggregate();
	void ScanLineOptimize();
	void MultiStepRefine();
};
