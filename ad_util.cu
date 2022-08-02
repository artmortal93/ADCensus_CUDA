#include "ad_util.cuh"

__device__ uint8_t hamming_distance(uint64_t val1, uint64_t val2)
{
    uint64_t dist = 0;
    uint64_t val = val1 ^ val2;
	
	dist = __popcll(val);
    // Count the number of set bits
    return static_cast<uint8_t>(dist);
}


__device__ uint64_t cencus_transform_9_7_d(uint8_t* img, int x, int y, int height, int width)
{
	
	if (x < 4 || x >= (height - 4))
		return 0ull;
	if (y < 3 || y >= (width - 3))
		return 0ull;
	const uint8_t gray_center = img[Get2dIdx(x, y, height, width)];
	uint64_t census_val = 0ull;
	int count_sum = 0;
    #pragma unroll
	for (int r = -4; r <= 4; r++) {
		for (int c = -3; c <= 3; c++) {
			census_val <<= 1;
			const uint8_t gray = img[Get2dIdx(x+r,y+c,height,width)];
			if (gray < gray_center) {
				census_val += 1;
				count_sum += 1;
			}
		}
	}
	return census_val;
}


__global__ void census_transform_97(uint8_t* img, uint64_t* census_array, int height, int width)
{   
	//block size is 32
    #define BLK_SIZE 32
    #define BLK_HEIGHT 32+2*4
    #define BLK_WIDTH  32+2*3
	__shared__ uint8_t shared_mem[BLK_HEIGHT][BLK_WIDTH]; //this mem contain all the possible window
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	//mapping thread idx to original thread first
	bool is_valid = (idx_x-4 < height) && (idx_y-3 < width) && (idx_x>=4) && (idx_y>=3);
	//load the first block first (top-left corner)
    ///https://stackoverflow.com/questions/38096915/cuda-convolution-mapping
	shared_mem[thread_idx][thread_idy] = is_valid?img[Get2dIdx(idx_x-4,idx_y-3,height,width)]:0;
	//__syncthreads();
	//load the second block bottom-left corner
	is_valid = (idx_x + 4 < height) && (idx_x+4>=0) && (idx_y - 3 >= 0) && (idx_y-3 <width);
	if (thread_idx >= (BLK_SIZE - 8))
	{
		shared_mem[thread_idx + 8][thread_idy] = is_valid?img[Get2dIdx(idx_x+4, idx_y-3,height, width)]:0;

	}
	//__syncthreads();
	//load the third block top-right corner
	is_valid = (idx_x-4>=0) && (idx_x-4<height) && (idx_y+3<width) && (idx_y+3>=0);
	if (thread_idy >= (BLK_SIZE - 6))
	{
		shared_mem[thread_idx][thread_idy+6] = is_valid? img[Get2dIdx(idx_x-4,idx_y+3,height, width)] : 0;
	}
	//__syncthreads();
	//load the fourth block bottom-right corner
	is_valid = (idx_x + 4 >= 0) && (idx_x + 4 < height) && (idx_y + 3 >= 0) && (idx_y + 3 < width);
	if (thread_idx >= (BLK_SIZE - 8) && thread_idy >= (BLK_SIZE - 6)) 
	{
		shared_mem[thread_idx+8][thread_idy+6]=is_valid? img[Get2dIdx(idx_x + 4, idx_y + 3, height, width)] : 0;
	}
	__syncthreads();
	//already load all the halo element now compute
	if (idx_x<height && idx_y<width) 
	{
		const uint8_t gray_center = shared_mem[thread_idx+4][thread_idy+3];//img[Get2dIdx(idx_x,idx_y,height,width)];
		uint64_t census_val = 0ull;
		int count_sum = 0;
        #pragma unroll
		for (int r = -4; r <= 4; r++) {
			for (int c = -3; c <= 3; c++) {
				census_val <<= 1;
				//bool outbound = (idx_x < 4 || idx_x >= (height - 4) || idx_y < 3 || idx_y >= (width - 3));
				const uint8_t gray = shared_mem[thread_idx + 4 + r][thread_idy + 3 + c];//outbound?0:img[Get2dIdx(idx_x+r, idx_y+c, height, width)];
				if (gray < gray_center) {
					census_val += 1;
					count_sum += 1;
				}
			}
		}
		bool outbound = (idx_x < 4 || idx_x >= (height - 4) || idx_y < 3 || idx_y >= (width - 3));
		
		census_array[Get2dIdx(idx_x, idx_y, height, width)] = outbound?0:census_val;
	}
}

__global__ void compute_gray(uint8_t* color_img, uint8_t* gray_img, int height, int width)
{   
	
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx_x < height && idx_y < width) {
		uchar3 rgb = FETCH_UCHAR3(color_img[Get2dIdxRGB(idx_x, idx_y, height, width)]);
		uint8_t gray = rgb.x * 0.299f + rgb.y * 0.587f + rgb.z * 0.114f;
		gray_img[Get2dIdx(idx_x, idx_y, height, width)] = gray;
	}
}

__global__ void compute_cost(uint8_t* color_left,
	                         uint8_t* gray_left,
	                         uint8_t* color_right, 
	                         uint8_t* gray_right,
	                         float* cost_init, 
	                         uint64_t* census_left,
	                         uint64_t* census_right,
	                         ADCensus_Option option,
	                         int height,
	                         int width)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int min_disparity = option.min_disparity;
	int max_disparity = option.max_disparity;
	int disp_range = option.max_disparity - option.min_disparity;
	int lambda_ad = option.lambda_ad;
	int lambda_census = option.lambda_census;
	if ((idx_x >= height) || (idx_y >= width))
		return;
	uchar3 pixel_l = FETCH_UCHAR3(color_left[Get2dIdxRGB(idx_x, idx_y, height, width)]);
	//need to improve with share memory
	uint64_t census_val_l = census_left[Get2dIdx(idx_x, idx_y, height, width)];
	for (int d = min_disparity; d < max_disparity; d++) {
		float& cost = cost_init[Get3dIdx(idx_x,idx_y,d,height,width,disp_range)];
		int yr = idx_y - d; //the pixel corresponding y of this disparity of right image
		if (yr < 0 || yr >= width) {
			cost = 1.0f;
			continue;
		}
		
		uchar3 pixel_r = FETCH_UCHAR3(color_right[Get2dIdxRGB(idx_x, yr, height, width)]);
		float cost_ad = (abs(pixel_l.x - pixel_r.x) + abs(pixel_l.y - pixel_r.y) + abs(pixel_l.z - pixel_r.z)) / 3.0f;
		uint64_t census_val_r = census_right[Get2dIdx(idx_x,yr,height,width)];
		float cost_census = hamming_distance(census_val_l, census_val_r);
		
		cost = 1 - exp(-cost_ad / lambda_ad) + 1 - exp(-cost_census / lambda_census);
		///for debug usage,debug ok
		//if (idx_x == 0 && idx_y == 0 && d==15)
		//  printf("cost of %d,%d,%d : %f \n", idx_x,idx_y,d,cost_i);
	}
	
}





