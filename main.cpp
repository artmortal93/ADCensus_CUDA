#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <filesystem>
#include <vector>
#include <tuple>
#include "adcensus_stereo.h"

using namespace std;
using namespace std::filesystem;

//write disparity does not reflect correct disparity relations
void write_disparity(std::string filename, float* disps, int _height, int _width, int min_disparity, int max_disparity)
{
    uint8_t* vis_data=new uint8_t[size_t(_height) * _width];
    float min_disp = float(_width), max_disp = -float(_width);
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            const float disp = disps[i * _width + j];
            if (disp != INFINITY) {
                min_disp = std::fmin(min_disp, disp);
                max_disp = std::fmax(max_disp, disp);
            }
        }
    }
    std::cout << "write out disparity, min disp is:" << min_disp << " max disp is" << max_disp << std::endl;
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            const float disp = disps[i * _width + j];
            if (disp == INFINITY) {
                vis_data[i * _width + j] = 0;
            }
            else {
                vis_data[i * _width + j] = static_cast<uint8_t>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }
    //write w,h,c
    int result=stbi_write_png(filename.c_str(), _width, _height, 1, (void*)vis_data,0);
    delete[] vis_data;
}
//you need to download from middlebury website
std::tuple<std::string, std::string, std::string> retrieve_middlebury_old(int idx) {
    std::vector<std::string> name = { "Aloe", "Baby1", "Baby2", "Baby3", "Bowling1",
                                        "Bowling2", "Cloth1", "Cloth2", "Cloth3", "Cloth4",
                                        "Flowerpots","Lampshade1", "Lampshade2", "Midd1", "Midd2",
                                        "Monopoly","Plastic", "Rocks1", "Rocks2", "Wood1",
                                        "Wood2" };
    auto imgdir_path = current_path() / "dataset";
    auto left_img_path = imgdir_path / name[idx] / "view1.png"; //disp1,disp5 in directory are ground true
    auto right_img_path = imgdir_path / name[idx] / "view5.png";
    return std::make_tuple(left_img_path.string(), right_img_path.string(),name[idx]);
}



std::tuple<std::string, std::string,std::string> retrieve_middlebury_latest(int idx) {
 std::vector<std::string> name = { "artroom1", "artroom2","bandsaw1","bandsaw2","chess1",
                                   "chess2","chess3","curule1","curule2","curule3",
                                   "ladder1","ladder2","octogons1","octogons2","pendulum1",
                                   "pendulum2","podium1","skates1","skates2","skiboots1",
                                   "skiboots2","skiboots3","traproom1","traproom2" };
     auto imgdir_path = current_path() / "dataset2";
     auto left_img_path = imgdir_path / name[idx] / "im0.png"; //disp1,disp5 in directory are ground true
     auto right_img_path = imgdir_path / name[idx] / "im1.png";
     return std::make_tuple(left_img_path.string(), right_img_path.string(),name[idx]);
}


int main()
{   
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    //change if it's needed
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL * 2048); //2GB
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda set heap size failed!");
        return 1;
    }
    int idx=23;
    auto path_tuple = retrieve_middlebury_latest(idx);
    auto left_img_path_str = std::get<0>(path_tuple);
    auto right_img_path_str = std::get<1>(path_tuple);
    //std::cout << "left path:" << std::filesystem::exists(left_img_path) << " " << left_img_path.string() << std::endl;
    //std::cout << "right path:" << std::filesystem::exists(right_img_path) << " " << right_img_path.string() << std::endl;
    auto scene_name = std::get<2>(path_tuple);
    std::replace(left_img_path_str.begin(), left_img_path_str.end(), '\\', '\/');
    std::replace(right_img_path_str.begin(), right_img_path_str.end(), '\\', '\/');
    int l_width, l_height, l_bpp;
    int r_width, r_height, r_bpp;
    uint8_t* image_left = stbi_load(left_img_path_str.c_str(), &l_width, &l_height, &l_bpp, 3);
    std::cout << "image width: " << l_width << "image height: " << l_height << "bpp: " << l_bpp << std::endl;
    uint8_t* image_right = stbi_load(right_img_path_str.c_str(), &r_width, &r_height, &r_bpp, 3);
    std::cout << "image width: " << r_width << "image height: " << r_height << "bpp: " << r_bpp << std::endl;
    std::cout << scene_name << std::endl;
    ADCensus_Option option;
    //the min disparity, must set to 0
    option.cross_L1 = option.cross_L1 * 2;
    option.cross_L2 = option.cross_L2 * 2;
    option.cross_t1 = option.cross_t1 * 2;
    option.cross_t2 = option.cross_t2 * 2;
    option.min_disparity = 0;
    //the max disparity, must set to times of 32 ,e.g. 64/128/160/256....
    //you always set the middlebury-new to 256, middlebury old to 128 will be best
    option.max_disparity = 256;
    //if not do filling, algorthim only do lr check, and will not fill the lr area, good for realtime application
    //if do filling, fill all the bad lr area base on paper's description
    option.do_filling = false;
    option.height = l_height;
    option.width = l_width;
    //threshold of lr check, if you want disparity map to be very precise, you could use 1.0, use high lrcheck could have smoother surface
    option.lrcheck_thres = 5.0f;
    ADCensusStereo* stereo = new ADCensusStereo(option);
    stereo->SetOption(option);
    stereo->Init();
    stereo->SetComputeImg(image_left, image_right);
    stereo->Compute();
    float* image_disp_left = stereo->RetrieveLeftDisparity();
    write_disparity(scene_name+"disp_left.png",image_disp_left,option.height,option.width,option.min_disparity,option.max_disparity);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    delete stereo;
    cudaDeviceSynchronize();
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;

}







  


