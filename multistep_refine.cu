#include "multistep_refine.cuh"
#include <assert.h>

__global__ void retrieve_left_disp(float* cost, float* disp_left, ADCensus_Option option)
{
	//double the shared mem for d
	extern __shared__ float smem[];
	int idx_x = blockIdx.x;
	int idx_y = blockIdx.y;
	int idx_d = threadIdx.x;
	int tid = idx_d;
	int height = option.height;
	int width = option.width;
	int disp_range = option.max_disparity - option.min_disparity;
	float* cost_displine = smem;
	float* cost_min_idx = (float*)&smem[disp_range];
	cost_displine[idx_d] = cost[Get3dIdx(idx_x, idx_y, idx_d, height, width, disp_range)];
	cost_min_idx[idx_d] = idx_d;
	__syncthreads();
	//record the cost of your neightbor pixel 
	float cost_1 = (idx_d - 1 >= 0) ? cost_displine[idx_d - 1] : 0;
	float cost_2 = (idx_d < disp_range - 1) ? cost_displine[idx_d + 1] : 0;
	
	for (int t = disp_range / 2; t >= 1; t=t / 2)
	{
		if (tid < t)
		{
			if (cost_displine[tid] > cost_displine[tid + t])
			{
				cost_min_idx[tid] = cost_min_idx[tid + t];
				cost_displine[tid] = cost_displine[tid + t];

			}
		}
		__syncthreads();
	}
	if (tid == cost_min_idx[0])
	{
		if (cost_min_idx[0] == option.min_disparity || cost_min_idx[0] >= option.max_disparity - 1)
		{
			disp_left[Get2dIdx(idx_x, idx_y, height, width)] = INVALID_FLOAT;
		}
		
		else {
			float min_cost = cost_displine[0];
			float denom = cost_1 + cost_2 - 2 * min_cost;
			float val = 0.0f;
			float epsilon = 0.00001f;
			////
			////originally you only check if the denom is equal to 0.0f but gpu has different percision than cpu
			////so you need a small epsilon to deal with this problem
			////you could change to 0.0f to see the result is totally different as two reasons
		    ////in boundary the cost are quite similar to each other
			////the system is very sensitive to numerical error
			if (denom <= epsilon)
			{
				val= float(cost_min_idx[0]);
			}
			else {
				val = float(cost_min_idx[0]) + (cost_1 - cost_2) / (denom * 2.0f);
				if (val <= 0.0f)
					val = float(cost_min_idx[0]);
			}
			disp_left[Get2dIdx(idx_x, idx_y, height, width)] = val;
		}
	}	
}

__global__ void retrieve_right_disp(float* cost, float* disp_left, ADCensus_Option option)
{
	extern __shared__ float smem[];
	int idx_x = blockIdx.x;
	int idx_y = blockIdx.y;
	int idx_d = threadIdx.x;
	int tid = idx_d;
	int height = option.height;
	int width = option.width;
	int disp_range = option.max_disparity - option.min_disparity;
	float* cost_displine = smem;
	float* cost_min_idx = (float*)&smem[disp_range];
	int col_left = idx_d + idx_y;
	if (col_left >= 0 && col_left < width)
	{
		cost_displine[idx_d] = cost[Get3dIdx(idx_x, col_left, idx_d, height, width, disp_range)];
	}
	else {
		cost_displine[idx_d] = 99999.0f;
	}
	cost_min_idx[idx_d] = idx_d;
	__syncthreads();
	float cost_1 = (idx_d - 1 >= 0) ? cost_displine[idx_d - 1] : 0;
	float cost_2 = (idx_d < disp_range - 1) ? cost_displine[idx_d + 1] : 0;
	for (int t = disp_range / 2; t >= 1; t=t / 2)
	{
		if (tid < t)
		{
			if (cost_displine[idx_d] > cost_displine[idx_d + t])
			{
				cost_min_idx[idx_d] = cost_min_idx[idx_d + t];
				cost_displine[idx_d] = cost_displine[idx_d + t];

			}
		}
		__syncthreads();
	}
	if (tid == cost_min_idx[0])
	{
		if (cost_min_idx[0] == option.min_disparity || cost_min_idx[0] == option.max_disparity - 1)
		{
			disp_left[Get2dIdx(idx_x, idx_y, height, width)] = cost_min_idx[0];
		}
		else {
			float min_cost = cost_displine[0];
			const float denom = cost_1 + cost_2 - 2 * min_cost;
			float val = 0.0f;
			float epsilon = 0.00001f;
			////
			////originally you only check if the denom is equal to 0.0f but gpu has different percision than cpu
			////so you need a small epsilon to deal with this problem
			////you could change to 0.0f to see the result is totally different as two reasons
			////in boundary the cost are quite similar to each other
			////the system is very sensitive to numerical error
			if (denom <= epsilon)
			{
				val = float(cost_min_idx[0]);
			}
			else {
				val = float(cost_min_idx[0]) + (cost_1 - cost_2) / (denom * 2.0f);
				if (val <= 0.0f)
					val = float(cost_min_idx[0]);
			}
			disp_left[Get2dIdx(idx_x, idx_y, height, width)] = val;
		}
	}	
}




//only mark all the invalid position on mask, but not to modify the original disp to prevent shared data collision
__global__ void outlier_detection(float* disp_left, float* disp_right, float* disp_out, uint8_t* disp_mask, ADCensus_Option option)
{   
	int width = option.width;
	int height = option.height;
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	float threshold = option.lrcheck_thres;
	if (x_idx < height && y_idx < width)
	{
		float disp = disp_left[Get2dIdx(x_idx, y_idx, height, width)];
		if (disp == INVALID_FLOAT) {
			disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = MISMATCHES;
			disp_out[Get2dIdx(x_idx, y_idx, height, width)] = INVALID_FLOAT;
			return;
		}
		else {
			float col_right = lround(y_idx - disp); //get the corresponding pixel on right disp map
			if (col_right >= 0 && col_right < width)
			{
				float disp_r = disp_right[Get2dIdx(x_idx, col_right, height, width)];
				if (abs(disp_r - disp) > threshold)
				{
					const int col_rl = lround(col_right + disp_r); //get the corresponding right pixel's left pixel
					//map back the disparity
					if (col_rl > 0 && col_rl < width)
					{
						float disp_l = disp_left[Get2dIdx(x_idx, col_rl, height, width)];
						disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = (disp_l > disp) ? OCCLUSIONS : MISMATCHES;
						disp_out[Get2dIdx(x_idx, y_idx, height, width)] = INVALID_FLOAT;
					}
					else {
						disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = MISMATCHES;
						disp_out[Get2dIdx(x_idx, y_idx, height, width)] = INVALID_FLOAT;
					}
				}
				else 
				{
					
					disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = NORMAL;
					disp_out[Get2dIdx(x_idx, y_idx, height, width)] = disp;
				}
			}
			else 
			{   
				disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = MISMATCHES;	
				disp_out[Get2dIdx(x_idx, y_idx, height, width)] = INVALID_FLOAT;
			}
		}
	}
}



__global__ void region_voting(float* disp_left, float* disp_out, uint8_t* disp_mask, CrossArm* arms, ADCensus_Option option)
{
	int width = option.width;
	int height = option.height;
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int disp_min = option.min_disparity;
	int disp_max = option.max_disparity;
	int disp_range = disp_max - disp_min;

	if (idx_x < height && idx_y < width)
	{
		int irv_ts = option.irv_ts;
		float irv_th = option.irv_th;
		uint8_t mask_val = disp_mask[Get2dIdx(idx_x, idx_y, height, width)];
		//only handle possible disp that violate the 
		if (mask_val == NORMAL)
		{
			disp_out[Get2dIdx(idx_x, idx_y, height, width)] = disp_left[Get2dIdx(idx_x, idx_y, height, width)];
			return;
		}
		int* histogram = nullptr;
		histogram = (int*)malloc(sizeof(int) * disp_range);
		assert(histogram != NULL);
		memset(histogram, 0, sizeof(int) * disp_range);
		CrossArm arm = arms[Get2dIdx(idx_x, idx_y, height, width)];
		for (int i = -arm.top; i <= arm.bottom; i++)
		{
			int x_t = idx_x + i;
			CrossArm arm2= arms[Get2dIdx(x_t, idx_y, height, width)];
			for (int j = -arm2.left; j < arm2.right; j++) {
				int y_t = idx_y + j;
				float d = disp_left[Get2dIdx(x_t, y_t, height, width)];
				//uint8_t temp_mask=disp_mask[Get2dIdx(x_t, y_t, height, width)];
				//remind that mask may be modified before other thread,only val in disp_left stay constant
				if (d!=INVALID_FLOAT)
					histogram[lround(d)]+=1;
			}
		}

		int best_disp = 0;
		int count = 0;
		int max_ht = 0;
		for (int d = 0; d < disp_range; d++) {
			const auto& h = histogram[d];
			if (max_ht < h) {
				max_ht = h;
				best_disp = d;
			}
			count += h;
		}
		//only if max_ht>0 could erase this pixel
		if (max_ht > 0) {
			if (count > irv_ts&& max_ht * 1.0f / count > irv_th) {
				float disp = best_disp + disp_min;
				disp_out[Get2dIdx(idx_x, idx_y, height, width)]=disp;
				disp_mask[Get2dIdx(idx_x, idx_y, height, width)] =NORMAL;
			}
			else {
				disp_out[Get2dIdx(idx_x, idx_y, height, width)] = disp_left[Get2dIdx(idx_x, idx_y, height, width)];
				disp_mask[Get2dIdx(idx_x, idx_y, height, width)] = mask_val;
			}
		}
		else{
			disp_out[Get2dIdx(idx_x, idx_y, height, width)] = disp_left[Get2dIdx(idx_x, idx_y, height, width)];
			disp_mask[Get2dIdx(idx_x, idx_y, height, width)] = mask_val;
		}
		free(histogram);
	}
}

__global__ void interpolation(float* disp_left, float* disp_out, uint8_t* img_left, uint8_t* disp_mask, ADCensus_Option option)
{    
	//each block handle a best color
	
	static constexpr int direction = 16;
	//per direction a block
	__shared__ float per_direction_best_color_dist[direction];
	__shared__ float per_direction_best_color_disp[direction];

	uchar3 current_color;
	float current_disp;
	const float pi = 3.1415926f;
	int min_disp = option.min_disparity;
	int max_disp = option.max_disparity;
	const int search_range = max_disp - min_disp;
	int width = option.width;
	int height = option.height;
	int x_idx = blockIdx.x;
	int y_idx = blockIdx.y;
	int thread_idx = threadIdx.x;
	int mask_val = disp_mask[Get2dIdx(x_idx, y_idx, height, width)];


	if (mask_val == NORMAL)
	{   
		if(thread_idx==0)
		    disp_out[Get2dIdx(x_idx, y_idx, height, width)]=disp_left[Get2dIdx(x_idx, y_idx, height, width)];
		return;
	}
	current_disp = disp_left[Get2dIdx(x_idx, y_idx, height, width)];
	per_direction_best_color_disp[thread_idx] = current_disp; //must be INVALID_FLOAT
	__syncthreads();
	
	current_color = FETCH_UCHAR3(img_left[Get2dIdxRGB(x_idx, y_idx, height, width)]);
	
	float min_occ_disp = 99999; //for occlustion
	int min_color_dist = 99999; //for mismatch,current direction best dist
	float min_color_disp = 99999; //for mismatch,current direction best disp
    

	double ang = threadIdx.x==0? 0: pi / float(threadIdx.x);
	const auto sina = sin(ang);
	const auto cosa = cos(ang);
	for (int m = 0; m < search_range; m++)
	{
		int xx = lround(x_idx + m * sina);
		int yy = lround(y_idx + m * cosa);
		if (yy >= 0 && yy < width && xx >= 0 && xx < height)
		{    
			//mismatch logic
			if (mask_val == MISMATCHES){
				uchar3 track_color = FETCH_UCHAR3(img_left[Get2dIdxRGB(xx, yy, height, width)]);
				float track_disp = disp_left[Get2dIdx(xx, yy, height, width)];
				int current_dist = ColorDist(track_color, current_color);
				if (current_dist < min_color_dist && track_disp!=INVALID_FLOAT)
				{   
					min_color_dist = current_dist;
					min_color_disp = track_disp;
				}
			}
			//occlusion logic
			else {
				float track_disp = disp_left[Get2dIdx(xx, yy, height, width)];
				if (track_disp < min_occ_disp && track_disp!=INVALID_FLOAT)
					min_occ_disp = track_disp;
			}
		}
	}
	__syncthreads();
	if (mask_val == MISMATCHES) {
		per_direction_best_color_disp[threadIdx.x] = min_color_disp;
		per_direction_best_color_dist[threadIdx.x] = min_color_dist;
	}
	else if(mask_val == OCCLUSIONS)
	{
		per_direction_best_color_disp[threadIdx.x] = min_occ_disp;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		float best_disp = 99999.0f;
		float best_dist = 99999.0f;
		if (mask_val == MISMATCHES) {
			for (int i = 0; i < direction; i++)
			{
				if (per_direction_best_color_dist[i] < best_dist) {
					best_dist = per_direction_best_color_dist[i];
					best_disp = per_direction_best_color_disp[i];
				}
			}
		}
		else if(mask_val==OCCLUSIONS){
			for (int i = 0; i < direction; i++)
			{
				if (per_direction_best_color_disp[i] < best_disp)
					  best_disp = per_direction_best_color_disp[i];
			}
		}
		disp_out[Get2dIdx(x_idx, y_idx, height, width)] = (best_disp == 99999.0f) ? INVALID_FLOAT : best_disp;
		disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = (best_disp == 99999.0f) ? mask_val : NORMAL;
	}
}

__global__ void median_filter(float* disp_left, float* disp_out, int height, int width)
{    
    #define wnd_size 3  //i only implement a 3x3 filter because it's no so necessary to have a dynamic setting of this function,
	//you can use template to speed up instead
	int radius = wnd_size / 2;
	int actual_size = 0; //the actual size should consider in median filter
	int wnd_idx = 0;
	//int size = wnd_size * wnd_size;
	int x_idx = blockDim.x*blockIdx.x+threadIdx.x;
	int y_idx = blockDim.y*blockIdx.y+threadIdx.y;
	if (x_idx < height && y_idx < width)
	{
		float local_mem[wnd_size*wnd_size] = { 0 };
        #pragma unroll
		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++)
			{
				int x_coord = x_idx+i;
				int y_coord = y_idx+j;
				bool out_boundary = (x_coord < 0) || (y_coord < 0) || (x_coord >= height) || (y_coord >= width);
				if (out_boundary)
				{   
					local_mem[wnd_idx] = -100000.0; //very low to make it sort to first
					wnd_idx++;
				}
				else {
					local_mem[wnd_idx] = disp_left[Get2dIdx(x_coord,y_coord,height,width)];
					wnd_idx++;
					actual_size++;
				}

			}
		}
		float local_disp_val = local_mem[wnd_size * wnd_size / 2];

		//bubble sort
		for (int i = 0; i < wnd_size * wnd_size - 1; i++)
		{
			for (int j = 0; j < wnd_size * wnd_size - i - 1; j++)
			{
				if (local_mem[j] > local_mem[j + 1])
				{
					float temp = local_mem[j+1];
					local_mem[j + 1] = local_mem[j];
					local_mem[j] = temp;
				}
			}
		}
		float* valid_mem = (float*)&local_mem[wnd_size*wnd_size-actual_size];
		float median = 0;
		if (actual_size == 0)
		{
			disp_out[Get2dIdx(x_idx, y_idx, height, width)] = local_disp_val;
		}
		else {
			median = valid_mem[actual_size / 2];
			disp_out[Get2dIdx(x_idx, y_idx, height, width)] = median;
		}
	}
}
