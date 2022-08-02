#include "scanline_optimize.cuh"
#define FULL_MASK 0xffffffff


__global__ void scanline_optimize_left2right(uint8_t* img_left, uint8_t* img_right,float* cost_init, float* cost_aggr, ADCensus_Option option)
{   
	//allocate d size thread and d*2 size shared memory
	//one for last aggr row and one for current init row and one for minimum reduction
	extern volatile __shared__ float smem[];
	int direction = 1;
	const int width = option.width;  //xdim
	const int height = option.height; //ydim
	const int min_disp = option.min_disparity;
	const int max_disp = option.max_disparity;
	const float p1 = option.so_p1;
	const float p2 = option.so_p2;
	const float tso = option.so_tso;
	const int disp_range = max_disp - min_disp;
	float* last_aggr_row_buffer = (float*)smem; //shared mem array with disp range
	float* cost_init_row_buffer = (float*)&smem[disp_range];//shared mem array with disp range
	const int row_idx = blockIdx.x;
	int d_idx = threadIdx.x;
	int tid = d_idx;
	int wrap_id = tid / 32;  //location of current wrap
	int wrap_num = max_disp / 32;
	//if (row_idx < height)
	//{
		//first to handle first col first col all disparity of cost aggr do no need to compute
		int col_idx = 0;	
		last_aggr_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
		__syncthreads();
		//becuase first column dont need to do any aggregation
		cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = last_aggr_row_buffer[d_idx];
		uchar3 color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
		uchar3 color_last = color;
		///caculate minimum cost on the shared memory
		__syncthreads();
		float min_cost_last_path = last_aggr_row_buffer[d_idx];
		for (int offset = 16; offset > 0; offset /= 2)
			min_cost_last_path = min(__shfl_down_sync(FULL_MASK, min_cost_last_path, offset),min_cost_last_path);
		if (tid % 32 == 0)
			cost_init_row_buffer[wrap_id] = min_cost_last_path; //save wrap reduce result to corresponding wrap id
		__syncthreads();
		if (tid == 0) {
			for (int w = 1; w < wrap_num; w++)
				cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
		}
		__syncthreads();
		min_cost_last_path = cost_init_row_buffer[0]; ///boardcast to every thread in the block
		col_idx += direction;
		__syncthreads();
		///copy 1st row as buffer of current row start of iteration
		cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
		__syncthreads();
		//seems ok till here
		for (int j = 0; j < width - 1; j++)
		{
			color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
			uint8_t d1 = ColorDist(color, color_last);
			uint8_t d2 = d1;
			int yr = col_idx - d_idx - min_disp;
			if (yr > 0 && yr < width - 1)
			{
				uchar3 color_r= FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx, yr, height, width)]);
				uchar3 color_last_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx, yr-direction, height, width)]);
				d2 = ColorDist(color_r, color_last_r);
			}
			float P1(0.0f), P2(0.0f);
			if (d1 < tso && d2 < tso) {
				P1 = p1;
				P2 = p2;
			}
			else if (d1 < tso && d2 >= tso) {
				P1 = p1 / 4.0f;
				P2 = p2 / 4.0f;
			}
			else if (d1 >= tso && d2 < tso) {
				P1 = p1 / 4.0f;
				P2 = p2 / 4.0f;
			}
			else if (d1 >= tso && d2 >= tso) {
				P1 = p1 / 10.0f;
				P2 = p2 / 10.0f;
			}
			const float cost = cost_init_row_buffer[d_idx];
			//l1 is current pixel
			const float l1 = last_aggr_row_buffer[d_idx];//d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] : 99999.0f;
			//l2 is pixel on the left
			const float l2 = d_idx-1>=0?last_aggr_row_buffer[d_idx-1]+P1:99999.0f+P1;
			//l3 is pixel on the right
			const float l3 = d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] + P1 : 99999.0f + P1;//d_idx + 2 < max_disp ? last_aggr_row_buffer[d_idx + 2] : 99999.0f;
			const float l4 = min_cost_last_path + P2;
			float cost_s = cost + float(min(min(l1, l2), min(l3, l4)));
			cost_s /= 2.0f;
			__syncthreads();
			cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = cost_s; //update cost aggr
			last_aggr_row_buffer[d_idx] = cost_s;               //IMPORTANT:update cost aggr last aggr row buffer for next column
			//__syncthreads();
			//caculate the best minimum cost this round,re-use cost_init_row_buffer
			float min_cost_this_path = cost_s;
			for (int offset = 16; offset > 0; offset /= 2)
				min_cost_this_path = min(__shfl_down_sync(FULL_MASK, min_cost_this_path, offset), min_cost_this_path);
			if (tid % 32 == 0)
				cost_init_row_buffer[wrap_id] = min_cost_this_path; //save wrap reduce result to corresponding wrap id
			__syncthreads();
			if (tid == 0) {
				for (int w = 0; w < wrap_num; w++)
					cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
			}
			__syncthreads();
			min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block to replace the cost of min_cost_last_path
			col_idx += direction; //prepare to start next column
			__syncthreads();
			//dont need to update in last row iteration
			if (col_idx<width)  
			{
				cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
				color_last = color;
			}
			__syncthreads();
		}  //end for
	//}   
	//if (row_idx == 100 && d_idx == 0) 
	//	printf("scanline val left 2 right of %d , %d %d is %f \n", row_idx,100,d_idx,cost_aggr[Get3dIdx(row_idx, 100, d_idx, height, width, disp_range)]);
	//if (row_idx == 100 && d_idx == 100)
	//	printf("scanline val left 2 right of %d , %d %d is %f \n", row_idx, 100, d_idx, cost_aggr[Get3dIdx(row_idx, 100, d_idx, height, width, disp_range)]);
}

__global__ void scanline_optimize_right2left(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr, ADCensus_Option option)
{
	//allocate d size thread and d*2 size shared memory
	//one for last aggr row and one for current init row and one for minimum reduction
	extern volatile __shared__ float smem[];
	int direction = -1;
	const int width = option.width;  //xdim
	const int height = option.height; //ydim
	const int min_disp = option.min_disparity;
	const int max_disp = option.max_disparity;
	const float p1 = option.so_p1;
	const float p2 = option.so_p2;
	const float tso = option.so_tso;
	const int disp_range = max_disp - min_disp;
	float* last_aggr_row_buffer = (float*)smem; //shared mem array with disp range
	float* cost_init_row_buffer = (float*)&smem[disp_range];;//shared mem array with disp range
    int row_idx = blockIdx.x;
	int d_idx = threadIdx.x;
	int tid = d_idx;
	int wrap_id = tid / 32;  //location of current wrap
	int wrap_num = max_disp / 32;
	//if (row_idx < height)
	//{
		//first to handle first col first col all disparity of cost aggr do no need to compute
		int col_idx = width-1;
		last_aggr_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
		__syncthreads();
		cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = last_aggr_row_buffer[d_idx];
		uchar3 color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
		uchar3 color_last = color;
		//caculate minimum cost on the shared memory
		__syncthreads();
		float min_cost_last_path = last_aggr_row_buffer[d_idx];
		for (int offset = 16; offset > 0; offset /= 2)
			min_cost_last_path = min(__shfl_down_sync(FULL_MASK, min_cost_last_path, offset), min_cost_last_path);
		if (tid % 32 == 0)
			cost_init_row_buffer[wrap_id] = min_cost_last_path; //save wrap reduce result to corresponding wrap id
		__syncthreads();
		if (tid == 0) {
			for (int w = 0; w < wrap_num; w++)
				cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
		}
		__syncthreads();
		min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block
		col_idx += direction;
		__syncthreads();
		//copy 1st row as buffer of current row start of iteration
		cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
		__syncthreads();
		for (int j = 0; j < width - 1; j++)
		{
			color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
			uint8_t d1 = ColorDist(color, color_last);
			uint8_t d2 = d1;
			int yr = col_idx - d_idx - min_disp;
			if (yr > 0 && yr < width - 1)
			{
				uchar3 color_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx, yr, height, width)]);
				uchar3 color_last_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx, yr - direction, height, width)]);
				d2 = ColorDist(color_r, color_last_r);
			}
			float P1(0.0f), P2(0.0f);
			if (d1 < tso && d2 < tso) {
				P1 = p1;
				P2 = p2;
			}
			else if (d1 < tso && d2 >= tso) {
				P1 = p1 / 4.0f;
				P2 = p2 / 4.0f;
			}
			else if (d1 >= tso && d2 < tso) {
				P1 = p1 / 4.0f;
				P2 = p2 / 4.0f;
			}
			else if (d1 >= tso && d2 >= tso) {
				P1 = p1 / 10.0f;
				P2 = p2 / 10.0f;
			}
			const float cost = cost_init_row_buffer[d_idx];
			const float l1 = last_aggr_row_buffer[d_idx];//d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] : 99999.0f;
			//l2 is pixel on the left
			const float l2 = d_idx - 1 >= 0 ? last_aggr_row_buffer[d_idx - 1] + P1 : 99999.0f + P1;
			//l3 is pixel on the right
			const float l3 = d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] + P1 : 99999.0f + P1;//d_idx + 2 < max_disp ? last_aggr_row_buffer[d_idx + 2] : 99999.0f;
			const float l4 = min_cost_last_path + P2;
			float cost_s = cost + float(min(min(l1, l2), min(l3, l4)));
			cost_s /= 2.0f;
			__syncthreads();
			cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = cost_s; //update cost aggr
			last_aggr_row_buffer[d_idx] = cost_s;               //IMPORTANT:update cost aggr last aggr row buffer for next column
			//caculate the best minimum cost this round,re-use cost_init_row_buffer
			float min_cost_this_path = cost_s;
			for (int offset = 16; offset > 0; offset /= 2)
				min_cost_this_path = min(__shfl_down_sync(FULL_MASK, min_cost_this_path, offset), min_cost_this_path);
			if (tid % 32 == 0)
				cost_init_row_buffer[wrap_id] = min_cost_this_path; //save wrap reduce result to corresponding wrap id
			__syncthreads();
			if (tid == 0) {
				for (int w = 0; w < wrap_num; w++)
					cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
			}
			__syncthreads();
			min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block to replace the cost of min_cost_last_path
			col_idx += direction; //prepare to start next column
			__syncthreads();
			//dont need to do this in last iteration
			if (col_idx >= 0) {
				cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
				color_last = color;
			}
			__syncthreads();
		} //end for loop
		//if (row_idx == 100 && d_idx == 0)
		//	printf("scanline val left 2 right of %d , %d %d is %f \n", row_idx, 100, d_idx, cost_aggr[Get3dIdx(row_idx, 100, d_idx, height, width, disp_range)]);
		//if (row_idx == 100 && d_idx == 100)
		//	printf("scanline val left 2 right of %d , %d %d is %f \n", row_idx, 100, d_idx, cost_aggr[Get3dIdx(row_idx, 100, d_idx, height, width, disp_range)]);
}

__global__ void scanline_optimize_top2bottom(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr, ADCensus_Option option)
{
	//allocate d size thread and d*2 size shared memory
	//one for last aggr row and one for current init row and one for minimum reduction
	extern volatile __shared__ float smem[];
	const int direction = 1; //from top 2 bottom index increase, increase row_idx
	const int width = option.width;  //xdim
	const int height = option.height; //ydim
	const int min_disp = option.min_disparity;
	const int max_disp = option.max_disparity;
	const float p1 = option.so_p1;
	const float p2 = option.so_p2;
	const float tso = option.so_tso;
	const int disp_range = max_disp - min_disp;
	float* last_aggr_row_buffer = (float*)smem; //shared mem array with disp range
	float* cost_init_row_buffer = (float*)&last_aggr_row_buffer[disp_range];//shared mem array with disp range
	const int col_idx = blockIdx.x; //move in row order, col stay cool
	int d_idx = threadIdx.x;
	int tid = d_idx;
	int wrap_id = tid / 32;  //location of current wrap
	int wrap_num = max_disp / 32;
		//first to handle first col first col all disparity of cost aggr do no need to compute
	int row_idx = 0;
	last_aggr_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
    __syncthreads();
	cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = last_aggr_row_buffer[d_idx];
	uchar3 color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
	uchar3 color_last = color;
	//caculate minimum cost on the shared memory
	__syncthreads();
	float min_cost_last_path = last_aggr_row_buffer[d_idx];
	for (int offset = 16; offset > 0; offset /= 2)
		min_cost_last_path = min(__shfl_down_sync(FULL_MASK, min_cost_last_path, offset), min_cost_last_path);
	if (tid % 32 == 0)
		cost_init_row_buffer[wrap_id] = min_cost_last_path; //save wrap reduce result to corresponding wrap id
	__syncthreads();
	if (tid == 0) {
		for (int w = 1; w < wrap_num; w++)
			cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
		}
	__syncthreads();
	min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block
	row_idx += direction;
	__syncthreads();
	//copy 1st row as buffer of current row start of iteration
	cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
	__syncthreads();
		
	for (int j = 0; j < height - 1; j++)
		{
			color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
			uint8_t d1 = ColorDist(color, color_last);
			uint8_t d2 = d1;
			int yr = col_idx - d_idx - min_disp;
			if (yr > 0 && yr < width - 1)
			{
				uchar3 color_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx, yr, height, width)]);
				uchar3 color_last_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx-direction, yr, height, width)]);
				d2 = ColorDist(color_r, color_last_r);
			}
			float P1(0.0f), P2(0.0f);
			if (d1 < tso && d2 < tso) {
				P1 = p1;
				P2 = p2;
			}
			else if (d1 < tso && d2 >= tso) {
				P1 = p1 / 4.0f;
				P2 = p2 / 4.0f;
			}
			else if (d1 >= tso && d2 < tso) {
				P1 = p1 / 4.0f;
				P2 = p2 / 4.0f;
			}
			else if (d1 >= tso && d2 >= tso) {
				P1 = p1 / 10.0f;
				P2 = p2 / 10.0f;
			}
			const float cost = cost_init_row_buffer[d_idx];
			const float l1 = last_aggr_row_buffer[d_idx];//d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] : 99999.0f;
			//l2 is pixel on the left
			const float l2 = d_idx - 1 >= 0 ? last_aggr_row_buffer[d_idx - 1] + P1 : 99999.0f + P1;
			//l3 is pixel on the right
			const float l3 = d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] + P1 : 99999.0f + P1;//d_idx + 2 < max_disp ? last_aggr_row_buffer[d_idx + 2] : 99999.0f;
			const float l4 = min_cost_last_path + P2;
			float cost_s = cost + float(min(min(l1, l2), min(l3, l4)));
			cost_s /= 2.0f;
			__syncthreads();
			cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = cost_s; //update cost aggr
			last_aggr_row_buffer[d_idx] = cost_s;               //IMPORTANT:update cost aggr last aggr row buffer for next column
			//caculate the best minimum cost this round,re-use cost_init_row_buffer
			float min_cost_this_path = cost_s;
			for (int offset = 16; offset > 0; offset /= 2)
				min_cost_this_path = min(__shfl_down_sync(FULL_MASK, min_cost_this_path, offset), min_cost_this_path);
			if (tid % 32 == 0)
				cost_init_row_buffer[wrap_id] = min_cost_this_path; //save wrap reduce result to corresponding wrap id
			__syncthreads();
			if (tid == 0) {
				for (int w = 1; w < wrap_num; w++)
					cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
			}
			__syncthreads();
			min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block to replace the cost of min_cost_last_path
			row_idx += direction; //prepare to start next row
			__syncthreads();
			//dont need to do this in last iteration
			if (row_idx < height) {
				cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
				color_last = color;
			}
			__syncthreads();
	    } //for loop ends
	//if (col_idx == 100 && d_idx == 0)
	//	printf("scanline val left 2 right of %d , %d %d is %f \n", 100, col_idx, d_idx, cost_aggr[Get3dIdx(100, col_idx, d_idx, height, width, disp_range)]);
	//if (col_idx == 100 && d_idx == 100)
	//	printf("scanline val left 2 right of %d , %d %d is %f \n", 100, col_idx, d_idx, cost_aggr[Get3dIdx(100, col_idx, d_idx, height, width, disp_range)]);
	
}

__global__ void scanline_optimize_bottom2top(uint8_t* img_left, uint8_t* img_right, float* cost_init, float* cost_aggr, ADCensus_Option option)
{
	//allocate d size thread and d*2 size shared memory
	//one for last aggr row and one for current init row and one for minimum reduction
	extern volatile __shared__ float smem[];
	int direction = -1; //from top 2 bottom index increase, increase row_idx
	const int width = option.width;  //xdim
	const int height = option.height; //ydim
	const int min_disp = option.min_disparity;
	const int max_disp = option.max_disparity;
	const float p1 = option.so_p1;
	const float p2 = option.so_p2;
	const float tso = option.so_tso;
	const int disp_range = max_disp - min_disp;
	float* last_aggr_row_buffer = (float*)smem; //shared mem array with disp range
	float* cost_init_row_buffer = (float*)&last_aggr_row_buffer[disp_range];//shared mem array with disp range
	const int col_idx = blockIdx.x; //move in row order, col stay cool
	int d_idx = threadIdx.x;
	int tid = d_idx;
	int wrap_id = tid / 32;  //location of current wrap
	int wrap_num = max_disp / 32;
	
		//first to handle first row first col all disparity of cost aggr do no need to compute
		int row_idx = height-1;
		last_aggr_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
		__syncthreads();
		cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = last_aggr_row_buffer[d_idx];
		uchar3 color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
		uchar3 color_last = color;
		//caculate minimum cost on the shared memory
		__syncthreads();
		float min_cost_last_path = last_aggr_row_buffer[d_idx];
		for (int offset = 16; offset > 0; offset /= 2)
			min_cost_last_path = min(__shfl_down_sync(FULL_MASK, min_cost_last_path, offset), min_cost_last_path);
		if (tid % 32 == 0)
			cost_init_row_buffer[wrap_id] = min_cost_last_path; //save wrap reduce result to corresponding wrap id
		__syncthreads();
		if (tid == 0) {
			for (int w = 1; w < wrap_num; w++)
				cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
		}
		__syncthreads();
		min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block
		row_idx += direction;
		__syncthreads();
		//copy 1st row as buffer of current row start of iteration
		cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
		__syncthreads();
		for (int j = 0; j < height - 1; j++)
		{
			color = FETCH_UCHAR3(img_left[Get2dIdxRGB(row_idx, col_idx, height, width)]);
			uint8_t d1 = ColorDist(color, color_last);
			uint8_t d2 = d1;
			int yr = col_idx - d_idx - min_disp;
			if (yr > 0 && yr < width - 1)
			{
				uchar3 color_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx, yr, height, width)]);
				uchar3 color_last_r = FETCH_UCHAR3(img_right[Get2dIdxRGB(row_idx - direction, yr, height, width)]);
				d2 = ColorDist(color_r, color_last_r);
			}
			float P1(0.0f), P2(0.0f);
			if (d1 < tso && d2 < tso) {
				P1 = p1;
				P2 = p2;
			}
			else if (d1 < tso && d2 >= tso) {
				P1 = p1 / 4;
				P2 = p2 / 4;
			}
			else if (d1 >= tso && d2 < tso) {
				P1 = p1 / 4;
				P2 = p2 / 4;
			}
			else if (d1 >= tso && d2 >= tso) {
				P1 = p1 / 10;
				P2 = p2 / 10;
			}
			const float cost = cost_init_row_buffer[d_idx];
			const float l1 = last_aggr_row_buffer[d_idx];
			//l2 is pixel on the left
			const float l2 = d_idx - 1 >= 0 ? last_aggr_row_buffer[d_idx - 1] + P1 : 99999.0f + P1;
			//l3 is pixel on the right
			const float l3 = d_idx + 1 < max_disp ? last_aggr_row_buffer[d_idx + 1] + P1 : 99999.0f + P1;
			const float l4 = min_cost_last_path + P2;
			float cost_s = cost + float(min(min(l1, l2), min(l3, l4)));
			cost_s /= 2.0f;
			__syncthreads();
			cost_aggr[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)] = cost_s; //update cost aggr
			last_aggr_row_buffer[d_idx] = cost_s;               //IMPORTANT:update cost aggr last aggr row buffer for next column
			//caculate the best minimum cost this round,re-use cost_init_row_buffer
			float min_cost_this_path = cost_s;
			for (int offset = 16; offset > 0; offset /= 2)
				min_cost_this_path = min(__shfl_down_sync(FULL_MASK, min_cost_this_path, offset), min_cost_this_path);
			if (tid % 32 == 0)
				cost_init_row_buffer[wrap_id] = min_cost_this_path; //save wrap reduce result to corresponding wrap id
			__syncthreads();
			if (tid == 0) {
				for (int w = 1; w < wrap_num; w++)
					cost_init_row_buffer[0] = min(cost_init_row_buffer[0], cost_init_row_buffer[w]); //use thread 0 to caculate final answer
			}
			__syncthreads();
			min_cost_last_path = cost_init_row_buffer[0]; //boardcast to every thread in the block to replace the cost of min_cost_last_path
			row_idx += direction; //prepare to start next column
			__syncthreads();
			//dont need to do this in last iteration
			if (row_idx >= 0) {
				cost_init_row_buffer[d_idx] = cost_init[Get3dIdx(row_idx, col_idx, d_idx, height, width, disp_range)];
				color_last = color;
			}
			__syncthreads();
		}
		//if (col_idx == 100 && d_idx == 0)
		//	printf("scanline val left 2 right of %d , %d %d is %f \n", 100, col_idx, d_idx, cost_aggr[Get3dIdx(100, col_idx, d_idx, height, width, disp_range)]);
		//if (col_idx == 100 && d_idx == 100)
		//	printf("scanline val left 2 right of %d , %d %d is %f \n", 100, col_idx, d_idx, cost_aggr[Get3dIdx(100, col_idx, d_idx, height, width, disp_range)]);
}
