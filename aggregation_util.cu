#include "aggregation_util.cuh"

#define MAX_ARM_LENGTH 255

__global__ void BuildArm(uint8_t* color_img_left, CrossArm* cross_arms, ADCensus_Option option)
{   
	//not using shared memory for too big halo size and dynamic search range,as L1/L2 could make the block have very large window
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	if (idx_x < height && idx_y < width) {
		CrossArm& cross_arm = cross_arms[Get2dIdx(idx_x,idx_y,height,width)];
		FindHorizontalArm(color_img_left, idx_x, idx_y, cross_arm.left, cross_arm.right,option);
		FindVerticalArm(color_img_left, idx_x, idx_y, cross_arm.top, cross_arm.bottom,option);
		//debug ok
	}	
}

__device__ void FindHorizontalArm(uint8_t* color_img_left, int x, int y, uint8_t& left, uint8_t& right,ADCensus_Option option)
{   
	auto L1 = option.cross_L1;
	auto L2 = option.cross_L2;
	auto t1 = option.cross_t1;
	auto t2 = option.cross_t2;
	auto height = option.height;
	auto width = option.width;
	//combine the memory fetching
	uchar3 color0 = FETCH_UCHAR3(color_img_left[Get2dIdxRGB(x, y, height, width)]);
	int temp_left = 0;
	int temp_right = 0;
	int dir = -1;
	//extend from two arm k=2,k=0 is left to right,k=1 is right to left
    #pragma unroll
	for (int k = 0; k < 2; k++) {
		int dir = k == 0 ? -1 : 1;
		uchar3 color_last = color0;
		int yn = y + dir;
		//search dir in maximum L1 distance
		for (int n = 0; n < L1; n++) {
			//boundary check
			bool out_boundary = (k == 0 && yn < 0) || (k==1 && yn>=width);
			if (out_boundary)
				break;
			
			uchar3 color = FETCH_UCHAR3(color_img_left[Get2dIdxRGB(x, yn, height, width)]);
			int color_dist1 = ColorDist(color, color0);
			if (color_dist1 >= t1)
				break;
			if (n > 0) {
				int color_dist2 = ColorDist(color, color_last);
				if (color_dist2 >= t1)
					break;
			}
			if (n + 1 > L2) {
				if (color_dist1 >= t2)
					break;
			}
			if (k == 0)
				temp_left += 1;
			else
				temp_right += 1;
			color_last = color;
			yn += dir;
		}
	}
	left = min(temp_left,MAX_ARM_LENGTH);
	right = min(temp_right,MAX_ARM_LENGTH);
}

__device__ void FindVerticalArm(uint8_t* color_img_left, int x, int y, uint8_t& top, uint8_t& bottom,ADCensus_Option option)
{
	auto L1 = option.cross_L1;
	auto L2 = option.cross_L2;
	auto t1 = option.cross_t1;
	auto t2 = option.cross_t2;
	auto height = option.height;
	auto width = option.width; uchar3 color0 = FETCH_UCHAR3(color_img_left[Get2dIdxRGB(x, y, height, width)]);
	int temp_top = 0;
	int temp_bottom = 0;
	int dir = -1;
#pragma unroll
	for (int k = 0; k < 2; k++) {
		int dir = k == 0 ? -1 : 1;
		uchar3 color_last = color0;
		int xn = x + dir;
		//search dir in maximum L1 distance
		for (int n = 0; n < L1; n++) {
			//boundary check
			bool out_boundary = (k == 0 && xn < 0) || (k == 1 && xn == height);
			if (out_boundary)
				break;

			uchar3 color = FETCH_UCHAR3(color_img_left[Get2dIdxRGB(xn, y, height, width)]);
			int color_dist1 = ColorDist(color, color0);
			if (color_dist1 >= t1)
				break;
			if (n > 0) {
				int color_dist2 = ColorDist(color, color_last);
				if (color_dist2 >= t1)
					break;
			}
			if (n + 1 > L2) {
				if (color_dist1 >= t2)
					break;
			}
			if (k == 0)
				temp_top += 1;
			else
				temp_bottom += 1;
			color_last = color;
			xn += dir;
		}
	}
	top = min(temp_top,MAX_ARM_LENGTH);
	bottom = min(temp_bottom,MAX_ARM_LENGTH);

}

__global__ void ComputeSubpixelCountHorizontal(CrossArm* cross_arms, uint16_t* count_buffer, ADCensus_Option option)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	if (idx_x < height && idx_y < width) {
		CrossArm& cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		uint16_t count = cross_arm.left + cross_arm.right+1;
		count_buffer[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
}

__global__ void ComputeSubpixelAggregateHorizontal(CrossArm* cross_arms, uint16_t* horizontal_count, uint16_t* count_buffer, ADCensus_Option option)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	if (idx_x < height && idx_y < width) {
		int count = 0;
		CrossArm& cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		for (int t = -cross_arm.top; t <= cross_arm.bottom; t++) {
			count += count_buffer[Get2dIdx(idx_x + t, idx_y, height, width)];
		}
		horizontal_count[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
}

__global__ void ComputeSubpixelCountHorizontalCooperativeGroup(CrossArm* cross_arms, uint16_t* horizontal_count, uint16_t* count_buffer, ADCensus_Option option)
{   
	auto g = this_grid();
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	int count = 0;
	CrossArm cross_arm;
	if (idx_x < height && idx_y < width) {
		cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		count = cross_arm.left + cross_arm.right+1;
		count_buffer[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
	g.sync(); //sync all grid to input the counter
	if (idx_x < height && idx_y < width) {
		count = 0;
		for (int t = -cross_arm.top; t <= cross_arm.bottom; t++) {
			count += count_buffer[Get2dIdx(idx_x + t, idx_y, height, width)];
		}
		horizontal_count[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
}

__global__ void ComputeSubpixelCountVertical(CrossArm* cross_arms, uint16_t* count_buffer, ADCensus_Option option)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	if (idx_x < height && idx_y < width) {
		CrossArm& cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		int count = cross_arm.top + cross_arm.bottom+1;
		count_buffer[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
}

__global__ void ComputeSubpixelAggregateVertical(CrossArm* cross_arms, uint16_t* horizontal_count, uint16_t* count_buffer, ADCensus_Option option)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	if (idx_x < height && idx_y < width) {
		int count = 0;
		CrossArm& cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		for (int t = -cross_arm.left; t <= cross_arm.right; t++) {
			count += count_buffer[Get2dIdx(idx_x, idx_y+t, height, width)];
		}
		horizontal_count[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
}

__global__ void ComputeSubpixelCountVerticalCooperativeGroup(CrossArm* cross_arms, uint16_t* vertical_count, uint16_t* count_buffer, ADCensus_Option option)
{
	auto g = this_grid();
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	int width = option.width;
	int height = option.height;
	int count = 0;
	CrossArm cross_arm;
	if (idx_x < height && idx_y < width) {
		cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		count = cross_arm.top + cross_arm.bottom + 1;
		count_buffer[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
	g.sync(); //sync all grid to input the counter
	if (idx_x < height && idx_y < width) {
		count = 0;
		for (int t = -cross_arm.left; t <= cross_arm.right; t++) {
			count += count_buffer[Get2dIdx(idx_x, idx_y+t, height, width)];
		}
		vertical_count[Get2dIdx(idx_x, idx_y, height, width)] = count;
	}
}

__global__ void AggregateInArms1stphase(float* cost_init, float* cost_buffer, CrossArm* cross_arms, bool horizontal_first, ADCensus_Option option)
{   
	//with 128/256 worker thread
	//extern __shared__ float buffer_cost[];
	int idx_d = threadIdx.x;
	int idx_x = blockIdx.x;
	int idx_y = blockIdx.y;
	int width = option.width;
	int height = option.height;
	int disp_range = option.max_disparity - option.min_disparity;
	float buffer_cost=0.0f;// cost_init[Get3dIdx(idx_x, idx_y, idx_d, height, width, disp_range)];
	//__syncthreads();
	if (idx_x < height && idx_y < width) {
		CrossArm cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		if (horizontal_first)
		{
			for (int i = -1*int(cross_arm.left); i <= int(cross_arm.right); i++)
			{
				buffer_cost += cost_init[Get3dIdx(idx_x, idx_y + i, idx_d, height, width, disp_range)];
			}
		}
		else {

			for (int i = -1*int(cross_arm.top); i <= int(cross_arm.bottom); i++)
			{
				buffer_cost += cost_init[Get3dIdx(idx_x + i, idx_y, idx_d, height, width, disp_range)];
			}

		}
		//__syncthreads();
		cost_buffer[Get3dIdx(idx_x, idx_y, idx_d, height, width, disp_range)] = buffer_cost;
	}
}

__global__ void AggregateInArms2ndphase(float* cost_init, float* cost_buffer, CrossArm* cross_arms, uint16_t* count, bool horizontal_first, ADCensus_Option option)
{   
	//with 128/256 worker thread
	//extern __shared__ float buffer_cost[];
	int idx_d = threadIdx.x;
	int idx_x = blockIdx.x;
	int idx_y = blockIdx.y;
	int width = option.width;
	int height = option.height;
	int disp_range = option.max_disparity - option.min_disparity;
	float buffer_cost = 0.0f;
	//__syncthreads();
	if (idx_x < height && idx_y < width) {
		CrossArm cross_arm = cross_arms[Get2dIdx(idx_x, idx_y, height, width)];
		//horizontal first then aggregate from top bottom in 2nd phase
		uint16_t sup_count = count[Get2dIdx(idx_x, idx_y, height, width)];
		if (horizontal_first)
		{
			for (int i = -1*int(cross_arm.top); i <= int(cross_arm.bottom); i++)
			{
				buffer_cost += cost_buffer[Get3dIdx(idx_x + i, idx_y, idx_d, height, width, disp_range)];
			}
			buffer_cost /= float(sup_count);
		}
		else {


			for (int i = -1*int(cross_arm.left); i <= int(cross_arm.right); i++)
			{
				buffer_cost += cost_buffer[Get3dIdx(idx_x, idx_y + i, idx_d, height, width, disp_range)];
			}
			buffer_cost /= float(sup_count);
		}
		cost_init[Get3dIdx(idx_x, idx_y, idx_d, height, width, disp_range)] = buffer_cost;//change the cost aggr back to cost init
	}
}




