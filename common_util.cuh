#pragma once
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


struct ADCensus_Option {
    int min_disparity;		// min disparity 0 only
    int	max_disparity;		// max disparity, only accept 64,128,256
    int width;
    int height;
    int	lambda_ad;			//ad color lambda
    int	lambda_census;		//census lambda
    int	cross_L1;			// 
    int  cross_L2;			// 
    int	cross_t1;			// 
    int  cross_t2;			// 
    float	so_p1;				// scanline parameter
    float	so_p2;				// scanline paramter 
    int	so_tso;				// scanline parameter tso
    int	irv_ts;				// Iterative Region Voting parameter ts
    float irv_th;				// Iterative Region Voting parameter th

    float	lrcheck_thres;		// lrcheck threshold

    bool	do_lr_check;					// 
    bool	do_filling;						// 

    __device__ __host__ ADCensus_Option() : min_disparity(0), max_disparity(64),
        width(128),
        height(256),
        lambda_ad(10), lambda_census(30),
        cross_L1(34), cross_L2(17),
        cross_t1(20), cross_t2(6),
        so_p1(1.0f), so_p2(3.0f),
        so_tso(15), irv_ts(20), irv_th(0.4f),
        lrcheck_thres(1.0f),
        do_lr_check(true), do_filling(true) {};

};



//represent a ad color pixel
struct AD_Color 
{
public:
	AD_Color()=default; //default constructor
	AD_Color(uint8_t r, uint8_t g, uint8_t b);
	AD_Color(const AD_Color& other); //copy constructor
	AD_Color& operator=(const AD_Color& other); //copy assignment
	
public:
	uint8_t r;
	uint8_t g;
	uint8_t b;
};

struct CrossArm {
    uint8_t left = 0u; 
    uint8_t right = 0u;
    uint8_t top = 0u;
    uint8_t bottom= 0u;
   
};


///x :row y:col
__device__ int Get3dIdx(int x, int y, int z, int xdim, int ydim, int zdim);


__device__ int Get3dIdxPitch(int x, int y, int z, int xdim, int ydim, int zdim,size_t pitch);


__device__ int Get2dIdx(int x, int y, int xdim, int ydim);

__device__ int Get2dIdxRGB(int x, int y, int xdim, int ydim);


__device__ int Get2dIdxPitch(int x, int y, int xdim, int ydim, size_t pitch);

__device__ int Get2dIdxPitchRGB(int x, int y, int xdim, int ydim, size_t pitch);


__device__ int ColorDist(uint8_t r0, uint8_t g0, uint8_t b0, uint8_t r1, uint8_t g1, uint8_t b1);


__device__ int ColorDist(AD_Color c0, AD_Color c1);


__device__ int ColorDist(uchar3 c0, uchar3 c1);