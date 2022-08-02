#include "common_util.cuh"


__device__ int Get3dIdx(int x, int y, int z, int xdim, int ydim, int zdim)
{
	int offset = x * ydim * zdim + y * zdim + z;
	return offset;	
}

__device__ int Get3dIdxPitch(int x, int y, int z, int xdim, int ydim, int zdim, size_t pitch)
{   
	//not implemented yet
	return 0;
}

__device__ int Get2dIdx(int x, int y, int xdim, int ydim)
{
	int offset = x * ydim + y;
	return offset;
}

__device__ int Get2dIdxRGB(int x, int y, int xdim, int ydim)
{
	int offset = x * ydim * 3 + y * 3;
	return offset;
}

__device__ int Get2dIdxPitch(int x, int y, int xdim, int ydim, size_t pitch)
{
	int offset = x * pitch + y;
	return offset;
}

__device__ int Get2dIdxPitchRGB(int x, int y, int xdim, int ydim, size_t pitch)
{   
	//not implemented yet
	return 0;
}

__device__ int ColorDist(uint8_t r0, uint8_t g0, uint8_t b0, uint8_t r1, uint8_t g1, uint8_t b1)
{
	return max(abs(r0 - r1), max(abs(g0 - g1), abs(b0 - b1)));
}

__device__ int ColorDist(AD_Color c0, AD_Color c1)
{
	return max(abs(c0.r - c1.r), max(abs(c0.g - c1.g), abs(c0.b - c1.b)));
}

__device__ int ColorDist(uchar3 c0, uchar3 c1)
{
	return max(abs(c0.x - c1.x), max(abs(c0.y - c1.y), abs(c0.z - c1.z)));
}

AD_Color::AD_Color(uint8_t r, uint8_t g, uint8_t b)
{
	this->r = r;
	this->g = g;
	this->b = b;
}

AD_Color::AD_Color(const AD_Color& other)
{
	this->r = other.r;
	this->g = other.g;
	this->b = other.b;
}

AD_Color& AD_Color::operator=(const AD_Color& other)
{    
	this->r = other.r;
	this->g = other.g;
	this->b = other.b;
	return *this;
	// TODO: insert return statement here
}
