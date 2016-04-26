#ifndef GPU_H
#define GPU_H
#include<CL/cl.h>
#include<iostream>
#include<string>
#include<fstream>
#include<cstdlib>

using namespace std;

class GpuOpenCl
{
public:
	GpuOpenCl(){};
	void GetPlatformIDs();
	void GetDeviceIDs();
	void CreateContext();
	void CreateCommandQueue();
	void CreateProgram(const char* kernelfile);
	void BuildProgram();
	void CreateKernelFern(const char* kernelname);
	void CreateMemBufferFern(unsigned char* img_buffer, int img_size, int* grid_buffer, int grid_size,unsigned char* features_buffer, int features_size, int* col_buffer);
	void SetKernelArgFern();
	void EnqueueTaskFern(int grid_size);
	void ReadBufferFern(int* fern_buffer, int fern_size);
	void ReleaseMemObjectFern();
	
	void CreateKernelMF(const char* kernelname);
	void CreateMemBufferMF(int* fern_buffer, int fern_size, float* posteriors_buffer, int posteriors_size,float* conf_buffer, int conf_size);
	void SetKernelArgMF();
	void EnqueueTaskMF(int grid_size);
	void ReadBufferMF(float* conf_buffer, int conf_size);
	void ReleaseMemObjectMF();

	void CreateKernelNCC(const char* kernelname);
	void CreateMemBufferNCC(float* patch_buffer, int patch_size,
																	float* pnEx_buffer, int pnEx_size,
																	int* pn_size_buffer,
																  float ncc_size);
	void SetKernelArgNCC();
	void EnqueueTaskNCC(int dim_0, int dim_1);
	void ReadBufferNCC(float* ncc_buffer, int ncc_size);
	void ReleaseMemObjectNCC();

	void CreateKernelCO(const char* kernelname);
	void CreateMemBufferCO(float* ncc_buffer, int ncc_size, int* pn_size_buffer, float conf_size);
	void SetKernelArgCO();
	void EnqueueTaskCO(int dim_0);
	void ReadBufferCO(float* conf_buffer, int conf_size);
	void ReleaseMemObjectCO();


private:
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernelFE;
	cl_kernel kernelMF;
	cl_kernel kernelGP;
	cl_kernel kernelNCC;
	cl_kernel kernelCO;

	cl_mem img_buf;
	cl_mem grid_buf;
	cl_mem features_buf;
	cl_mem fern_buf;
	cl_mem col_buf;

	cl_mem posteriors_buf;
	cl_mem conf_buf;

	cl_mem patch_buf;
	cl_mem pnEx_buf;
	cl_mem pn_size_buf;
	cl_mem ncc_buf;
	
	cl_mem sconf_buf;
};




#endif
