#include "adcensus_stereo.h"
#include "scanline_optimize.cuh"
#include "aggregation_util.cuh"
#include "ad_util.cuh"
#include "multistep_refine.cuh"



ADCensusStereo::~ADCensusStereo()
{
	FreeCudaResource();
}

void ADCensusStereo::CleanUpMemory()
{   
	if (!first_time) {
		//only clean up the memory part and transfer memory
		int disparity_min = m_option.min_disparity;
		int disparity_max = m_option.max_disparity;
		int disparity_range = disparity_max - disparity_min;
		//set memory to 0 (if neccessary,seems not..)
		gpuErrchk(cudaMemset(census_left_d, 0, width * height * sizeof(uint64_t)));
		gpuErrchk(cudaMemset(census_right_d, 0, width * height * sizeof(uint64_t)));
		gpuErrchk(cudaMemset(cost_aggr_d, 0, width * height * disparity_range * sizeof(float)));
		gpuErrchk(cudaMemset(cost_init_d, 0, width * height * disparity_range * sizeof(float)));
		gpuErrchk(cudaMemset(vec_counter_horizontal_d, 0, width * height * sizeof(uint16_t)));
		gpuErrchk(cudaMemset(vec_counter_vertical_d, 0, width * height * sizeof(uint16_t)));
		gpuErrchk(cudaMemset(vec_counter_buffer_d, 0, width * height * sizeof(uint16_t)));
		gpuErrchk(cudaMemset(disp_mask_d, 0, width * height * sizeof(uint8_t)));
		gpuErrchk(cudaMemset(disp_left_d, 0, width * height * sizeof(float)));
		gpuErrchk(cudaMemset(disp_left_buffer_d, 0, width * height * sizeof(float)));
		gpuErrchk(cudaMemset(disp_right_d, 0, width * height * sizeof(float)));
		
	}
}

void ADCensusStereo::AllocateCudaResource()
{
	int disparity_min = m_option.min_disparity;
	int disparity_max = m_option.max_disparity;
	int disparity_range = disparity_max - disparity_min;

	gpuErrchk(cudaMalloc((void**)&img_left_rgb_d, width * height * 3 * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&img_right_rgb_d, width * height * 3 * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&img_left_gray_d, width * height * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&img_right_gray_d, width * height * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&census_left_d, width * height * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc((void**)&census_right_d, width * height * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc((void**)&cost_aggr_d, (size_t)width * height * disparity_range * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&cost_init_d, (size_t)width * height * disparity_range * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&cross_arm_left_d, width * height * sizeof(CrossArm)));
	gpuErrchk(cudaMalloc((void**)&vec_counter_horizontal_d, width * height * sizeof(uint16_t)));
	gpuErrchk(cudaMalloc((void**)&vec_counter_vertical_d, width * height * sizeof(uint16_t)));
	gpuErrchk(cudaMalloc((void**)&vec_counter_buffer_d, width * height * sizeof(uint16_t)));
	gpuErrchk(cudaMalloc((void**)&disp_mask_d, width * height * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&disp_left_d, width * height * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&disp_left_buffer_d, width * height * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&disp_right_d, width * height * sizeof(float)));
	
	//set memory to 0 (if neccessary,seems not..)
	gpuErrchk(cudaMemset(census_left_d, 0, width * height * sizeof(uint64_t)));
	gpuErrchk(cudaMemset(census_right_d, 0, width * height * sizeof(uint64_t)));
	gpuErrchk(cudaMemset(cost_aggr_d, 0, (size_t)width * height * disparity_range * sizeof(float)));
	gpuErrchk(cudaMemset(cost_init_d, 0, (size_t)width * height * disparity_range * sizeof(float)));
	gpuErrchk(cudaMemset(vec_counter_horizontal_d, 0, width * height * sizeof(uint16_t)));
	gpuErrchk(cudaMemset(vec_counter_vertical_d, 0, width * height * sizeof(uint16_t)));
	gpuErrchk(cudaMemset(vec_counter_buffer_d, 0, width * height * sizeof(uint16_t)));
	gpuErrchk(cudaMemset(disp_mask_d, 0, width * height * sizeof(uint8_t)));
	gpuErrchk(cudaMemset(disp_left_d, 0, width * height * sizeof(float)));
	gpuErrchk(cudaMemset(disp_left_buffer_d, 0, width * height * sizeof(float)));
	gpuErrchk(cudaMemset(disp_right_d, 0, width * height * sizeof(float)));
	//transfer memory 


	disp_left_h = (float*)calloc((size_t)width * height, sizeof(float));
	disp_right_h = (float*)calloc((size_t)width * height, sizeof(float));
	//allocate multiple stream
	if (this->speedup_use_multiple_stream) 
	{
		for (int i = 0; i < stream_num; i++)
		{
			gpuErrchk(cudaStreamCreate(&streams[i]));
		}
	}

}

void ADCensusStereo::FreeCudaResource()
{
	gpuErrchk(cudaFree(img_left_gray_d));
	gpuErrchk(cudaFree(img_right_gray_d));
	gpuErrchk(cudaFree(img_left_rgb_d));
	gpuErrchk(cudaFree(img_right_rgb_d));
	gpuErrchk(cudaFree(census_left_d));
	gpuErrchk(cudaFree(census_right_d));
	gpuErrchk(cudaFree(cost_aggr_d));
	gpuErrchk(cudaFree(cost_init_d));
	gpuErrchk(cudaFree(cross_arm_left_d));
	gpuErrchk(cudaFree(vec_counter_horizontal_d));
	gpuErrchk(cudaFree(vec_counter_vertical_d));
	gpuErrchk(cudaFree(vec_counter_buffer_d));
	gpuErrchk(cudaFree(disp_mask_d));
	gpuErrchk(cudaFree(disp_left_d));
	gpuErrchk(cudaFree(disp_left_buffer_d));
	gpuErrchk(cudaFree(disp_right_d));

	for (int i = 0; i < stream_num; i++)
	{    
	   gpuErrchk(cudaStreamDestroy(streams[i]));
	}
	//free(streams);
	free(disp_left_h);
	free(disp_right_h);
}

void ADCensusStereo::Init()
{   
	if (first_time) {
		AllocateCudaResource();
		first_time = false;
	}
	else {
	    //do nothing
	}
}

void ADCensusStereo::Reset()
{
	CleanUpMemory();
}

void ADCensusStereo::SetComputeImg(uint8_t* left_img, uint8_t* right_img)
{
	this->img_left_rgb_h = left_img;
	this->img_right_rgb_h = right_img;
	gpuErrchk(cudaMemcpy(img_left_rgb_d, img_left_rgb_h, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(img_right_rgb_d, img_right_rgb_h, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

void ADCensusStereo::Compute()
{

	CostCompute(); //ok
	CostAggregate();//ok
	ScanLineOptimize();//ok, ok first two optimize
	MultiStepRefine();
}

float* ADCensusStereo::RetrieveLeftDisparity()
{   
	
	return disp_left_h;
}

float* ADCensusStereo::RetrieveRightDisparity()
{   
	return disp_right_h;
}




void ADCensusStereo::CostCompute()
{   
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	constexpr unsigned int BLOCK_SIZE = 32;
	unsigned int block_dim_x = height / BLOCK_SIZE + 1;
	unsigned int block_dim_y = width / BLOCK_SIZE + 1;
	dim3 blockconfig = { BLOCK_SIZE,BLOCK_SIZE,1 };
	dim3 gridconfig = { block_dim_x,block_dim_y,1 };
	if (this->speedup_use_multiple_stream) {	
		compute_gray <<< gridconfig, blockconfig, 0, streams[0] >>> (img_left_rgb_d, img_left_gray_d, height, width);
		compute_gray <<< gridconfig, blockconfig, 0, streams[1] >>> (img_right_rgb_d, img_right_gray_d, height, width);
		//gpuErrchk(cudaPeekAtLastError());
		census_transform_97 <<< gridconfig, blockconfig, 0, streams[0] >>> (img_left_gray_d,census_left_d,height,width);
		census_transform_97 <<< gridconfig, blockconfig, 0, streams[1]>>> (img_right_gray_d,census_right_d,height,width);
		for (int i = 0; i < 2; i++)
			cudaStreamSynchronize(streams[i]);
		//gpuErrchk(cudaPeekAtLastError());
		
		compute_cost <<< gridconfig, blockconfig, 0, streams[0] >>> (img_left_rgb_d,
			                                                         img_left_gray_d,
			                                                         img_right_rgb_d,
			                                                         img_right_gray_d,
			                                                         cost_init_d,
			                                                         census_left_d,
			                                                         census_right_d,
			                                                         m_option,
			                                                         height,
			                                                         width                                                
		                                                         );
         
		cudaStreamSynchronize(streams[0]);
		//gpuErrchk(cudaPeekAtLastError());
	}
	else {
		//synchornize using default null stream, no need to config more
		compute_gray <<< gridconfig, blockconfig >>> (img_left_rgb_d, img_left_gray_d, height, width);
		//gpuErrchk(cudaPeekAtLastError());
		compute_gray <<< gridconfig, blockconfig >>> (img_right_rgb_d, img_right_gray_d, height, width);
		//gpuErrchk(cudaPeekAtLastError());
		census_transform_97 <<< gridconfig, blockconfig >>> (img_left_gray_d, census_left_d, height, width);
		//gpuErrchk(cudaPeekAtLastError());
		census_transform_97 <<< gridconfig, blockconfig >>> (img_right_gray_d, census_right_d, height, width);
		//gpuErrchk(cudaPeekAtLastError());
		compute_cost <<< gridconfig, blockconfig>>> (img_left_rgb_d,
			img_left_gray_d,
			img_right_rgb_d,
			img_right_gray_d,
			cost_init_d,
			census_left_d,
			census_right_d,
			m_option,
			height,
			width
			);
		cudaDeviceSynchronize();
		gpuErrchk(cudaPeekAtLastError());
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Ellaped time of cost init:%f ms\n", milliseconds);
}

void ADCensusStereo::CostAggregate()
{   
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	constexpr unsigned int BLOCK_SIZE = 32;
	unsigned int block_dim_x = height / BLOCK_SIZE + 1;
	unsigned int block_dim_y = width / BLOCK_SIZE + 1;
	dim3 blockconfig = { BLOCK_SIZE,BLOCK_SIZE,1 };
	dim3 gridconfig = { block_dim_x,block_dim_y,1 };
	//In case your image is pretty small, you could try to use cooperative group, but you are responsible to caculate the maximum
	// capacity for the GPU
	BuildArm <<< gridconfig, blockconfig,0,streams[0] >>> (img_left_rgb_d,cross_arm_left_d,m_option);
	ComputeSubpixelCountHorizontal <<< gridconfig, blockconfig,0,streams[0]>>> (cross_arm_left_d,vec_counter_buffer_d,m_option);
	ComputeSubpixelAggregateHorizontal <<< gridconfig, blockconfig,0,streams[0]>> > (cross_arm_left_d,vec_counter_horizontal_d,vec_counter_buffer_d,m_option);
	ComputeSubpixelCountVertical <<< gridconfig, blockconfig,0,streams[0]>>> (cross_arm_left_d,vec_counter_buffer_d,m_option);
	ComputeSubpixelAggregateVertical <<< gridconfig, blockconfig,0,streams[0]>> > (cross_arm_left_d, vec_counter_vertical_d, vec_counter_buffer_d, m_option);
	cudaStreamSynchronize(streams[0]);
	gpuErrchk(cudaPeekAtLastError());
	
	unsigned int disp_range = m_option.max_disparity - m_option.min_disparity;
	dim3 worker_gridconfig = { height,width,1 };
	dim3 worker_blockconfig = { disp_range,1,1 };
    //horizontal_first,then vertical
	///iter1 horizontal
	bool graphCreated = false;
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	for (int it = 0; it < 2; it++)
	{
		if (!graphCreated) {
			cudaStreamBeginCapture(streams[0], cudaStreamCaptureMode::cudaStreamCaptureModeGlobal);
			AggregateInArms1stphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, true, m_option);
			AggregateInArms2ndphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, vec_counter_horizontal_d, true, m_option);
			///iter2 vertical
			//cudaStreamSynchronize(streams[0]);
			AggregateInArms1stphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, false, m_option);
			AggregateInArms2ndphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, vec_counter_vertical_d, false, m_option);
			cudaStreamEndCapture(streams[0], &graph);
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			graphCreated = true;
		}
	    cudaGraphLaunch(instance, streams[0]);
		cudaStreamSynchronize(streams[0]);
	}
	///iter3 horizontal
	//cudaStreamSynchronize(streams[0]);
	//AggregateInArms1stphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, true, m_option);
	//AggregateInArms2ndphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, vec_counter_horizontal_d, true, m_option);
	///iter4 vertical
	//cudaStreamSynchronize(streams[0]);
	//AggregateInArms1stphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, false, m_option);
	//AggregateInArms2ndphase << <worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (cost_init_d, cost_aggr_d, cross_arm_left_d, vec_counter_vertical_d, false, m_option);
	cudaStreamSynchronize(streams[0]); 
	//gpuErrchk(cudaPeekAtLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Ellaped time of aggregation:%f ms\n", milliseconds);
}

void ADCensusStereo::ScanLineOptimize()
{   
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//after 4iteration the cost of iterative refinement come back to cost_init_d, so start as cost_init_d
	unsigned int disp_range = m_option.max_disparity - m_option.min_disparity;
	dim3 worker_left2rightgridconfig = { height,1,1 };
	dim3 worker_top2bottomgridconfig = { width,1,1 };
	dim3 worker_blockconfig = { disp_range,1,1 };
	unsigned int shared_mem_usage = disp_range * 2*sizeof(float);
	scanline_optimize_left2right<<<worker_left2rightgridconfig, worker_blockconfig, shared_mem_usage, streams[0]>>>(img_left_rgb_d, img_right_rgb_d, cost_init_d, cost_aggr_d, m_option);
	scanline_optimize_right2left<<<worker_left2rightgridconfig, worker_blockconfig, shared_mem_usage, streams[0]>>>(img_left_rgb_d, img_right_rgb_d, cost_aggr_d, cost_init_d, m_option);
	scanline_optimize_top2bottom<<<worker_top2bottomgridconfig, worker_blockconfig, shared_mem_usage, streams[0]>>>(img_left_rgb_d, img_right_rgb_d, cost_init_d, cost_aggr_d, m_option);
	scanline_optimize_bottom2top<<<worker_top2bottomgridconfig, worker_blockconfig, shared_mem_usage, streams[0]>>>(img_left_rgb_d, img_right_rgb_d, cost_aggr_d, cost_init_d, m_option);
	cudaStreamSynchronize(streams[0]);
	//gpuErrchk(cudaPeekAtLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Ellaped time of scan line optimize:%f ms\n", milliseconds);
}

void ADCensusStereo::MultiStepRefine()
{   
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	unsigned int disp_range = m_option.max_disparity - m_option.min_disparity;
	unsigned int shared_mem_usage = disp_range * 2 * sizeof(float);
	dim3 worker_gridconfig = { height,width,1 };
	dim3 worker_blockconfig = { disp_range,1,1 };
	retrieve_left_disp <<< worker_gridconfig, worker_blockconfig,shared_mem_usage,streams[0] >>> (cost_init_d,disp_left_d,m_option);
	retrieve_right_disp <<< worker_gridconfig,worker_blockconfig,shared_mem_usage,streams[0] >>> (cost_init_d,disp_right_d, m_option);
	cudaStreamSynchronize(streams[0]);
	gpuErrchk(cudaPeekAtLastError());
	constexpr unsigned int BLOCK_SIZE = 16;
	unsigned int block_dim_x = height / BLOCK_SIZE + 1;
	unsigned int block_dim_y = width / BLOCK_SIZE + 1;
	dim3 blockconfig = { BLOCK_SIZE,BLOCK_SIZE,1 };
	dim3 gridconfig = { block_dim_x,block_dim_y,1 };
	dim3 interpolationgridconfig = {height,width,1};
	dim3 interpolationblockconfig = { 16,1,1 };
	if (m_option.do_filling) {
		outlier_detection << <gridconfig, blockconfig, 0, streams[0] >> > (disp_left_d, disp_right_d, disp_left_buffer_d, disp_mask_d, m_option);
		//do region voting 4 times
		
		region_voting << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_buffer_d, disp_left_d, disp_mask_d, cross_arm_left_d, m_option);
		region_voting << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_d, disp_left_buffer_d, disp_mask_d, cross_arm_left_d, m_option);
		region_voting << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_buffer_d, disp_left_d, disp_mask_d, cross_arm_left_d, m_option);
		region_voting << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_d, disp_left_buffer_d, disp_mask_d, cross_arm_left_d, m_option);
		//interpolation << < interpolationgridconfig, interpolationblockconfig, 0, streams[0] >> > (disp_left_buffer_d, disp_left_d, img_left_rgb_d, disp_mask_d, m_option);
		//median_filter << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_d, disp_left_buffer_d, height, width);
	}
	else {
		outlier_detection << <gridconfig, blockconfig, 0, streams[0] >> > (disp_left_d, disp_right_d, disp_left_buffer_d, disp_mask_d, m_option);
	}
	//median_filter << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_buffer_d, disp_left_d, height, width);
	cudaStreamSynchronize(streams[0]);
	gpuErrchk(cudaPeekAtLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Ellaped time of postprocessing:%f ms\n", milliseconds);
	float* disp_output_d = nullptr;
	disp_output_d = disp_left_buffer_d;
	gpuErrchk(cudaMemcpy(disp_left_h, disp_output_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(disp_right_h, disp_right_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaPeekAtLastError());
	/*
	uint8_t* disp_mask_h = (uint8_t*)malloc(sizeof(uint8_t) * height * width);
	cudaMemcpy(disp_mask_h, disp_mask_d, sizeof(uint8_t)*height*width,cudaMemcpyDeviceToHost);
	int occlusion_sum = 0;
	int mismatch_sum = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			uint8_t mask = disp_mask_h[i * width + j];
			if (mask == OCCLUSIONS)
				occlusion_sum += 1;
			if (mask == MISMATCHES)
				mismatch_sum += 1;
		}
	}
	*/
}
